from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv
import os
import sys
import json
import time
import logging
from datetime import datetime

load_dotenv()

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"test_{model_name}_{timestamp}.log")

    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized, log file: {log_filename}")
    return logger

logger = setup_logging()

url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model_name = "qwen3-235b-a22b-thinking-2507"

client = AsyncOpenAI(
    api_key=os.getenv("ALI_API_KEY"),
    base_url=url
)


MAX_CONCURRENT_REQUESTS = 25
REQUESTS_PER_MINUTE = 500
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 3.0

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

test_dataset_base_path = "datasets/test"
tasks = [
    'Multiclass_Product_Classification',
    'Product_Matching',
    'Product_Substitute_Identification',
    'Product_Relation_Prediction',
    'Answer_Generation',
    'Sequential_Recommendation'
]

def get_system_prompts_for_task(task):
    task_to_system_prompt = {
        'Product_Matching': "You are an expert in product matching, determine accurately using product information whether the two products are identical, then respond only with yes if they are identical or no if they are not, based solely on the detailed product information provided.",
        'Answer_Generation': "You are an expert in information synthesis, analyze the user's query to identify its core intent, extract relevant details from the provided document, generate accurate and contextually relevant answers based solely on its content, then explicitly respond only with those answers.",
        'Product_Substitute_Identification': "You are an expert in product substitute identification, determine whether the product is partially relevant to the query and can functionally substitute unmet requirements, using product and query information, then respond with yes if it can or no if it cannot.",
        'Product_Relation_Prediction': "You are an expert in E-commerce product relationship analysis, accurately analyze the titles of Product 1 and Product 2 to precisely select the option that clearly indicates their relationship using only title information, then explicitly respond strictly with the chosen option.",
        'Sequential_Recommendation': "You are an expert in sequential product recommendation, estimate the user's intent based on their purchase history, then predict which product they are most likely to purchase next from the given options, then respond only with the predicted product.",
        'Multiclass_Product_Classification': "You are an expert in multi-type product classification, compare the user query with the product title to determine if the product meets the query specifications then select the option that best describes their relevance, and respond only with the selected option."
    }
    return task_to_system_prompt.get(task, "You are a helpful assistant.")


def get_user_prompt_for_task(task, instruction, input_data):
    task_to_user_prompt = {
        'Product_Matching': """
task: Product Matching
instruction: {instruction}
input: {input_data}
output: <answer>yes</answer> or <answer>no</answer>

Example:
task: Product Matching
instruction: Analyze the title, description, manufacturer, and price between the two products below and generate an output of yes if the two products are the same, otherwise respond with no.
input: {"product 1": {"title": "poser 7 win 2000 xp mac os 10.4 & up", "description": "poser 7 helps you create more realistic and advanced 3d characters. output the human figure in multiple styles as well as in multiple file formats. the all-new character morphing tools aids artists in developing a more impressive more accurate human figure. use the camera controls like a film director --...", "manufacturer": "aladdin systems", "price": "199.99"}, "product 2": {"title": "the coasters yamaha the very best of the coasters - smart pianosoft", "description": "this innovative software series enables your disklavier mark iii piano to perform with the world's most popular cds! using yamaha's pianosmart technology this companion diskette will magically empower your disklavier mark iii to accompany the ...", "manufacturer": "nan", "price": "18.5"}}
reasoning:
1. Title comparison: Product 1 is "Poser 7," a 3D character creation software; Product 2 is "The Very Best of the Coasters," a Yamaha music software diskette. Titles show no overlap.
2. Description comparison: Product 1 emphasizes 3D modeling and morphing tools, while Product 2 describes piano accompaniment software tied to music CDs. Functions are entirely unrelated.
3. Manufacturer comparison: Product 1 lists Aladdin Systems; Product 2 lists none ("nan"). They differ completely in brand identity.
4. Price comparison: Product 1 costs $199.99, Product 2 costs $18.50, showing a wide disparity consistent with different categories.
5. Conclusion: The two products belong to completely different domains (3D modeling vs. music accompaniment), with no shared attributes.
output: <answer>no</answer>
""",
        'Answer_Generation': """
task: Answer Generation
instruction: {instruction}
input: {input_data}
output: <answer>accurate and contextually relevant answers based solely on the content of the provided document.</answer>

Example:
task: Answer Generation
instruction:Given a question and the related document, and generate the answer to the question based on the information provided in the document.
input: {"question": "I want to buy this mug and please, tell me if it has a taste of iron?", "document": ["I think it's great. My husband was thinking of the KleenKanteen mug but didn't want to tell me that...I like the way the lid snaps shut and can lock open. I also like having a handle on the side.", "This is an excellent mug for hot drinks. One of the things I like about it is that it is easy to clean. I drink both coffee and tea from it so I need to be able to clean it to keep the coffee taste and smell from ruining the tea. One important thing to note about the lid, this isn't the best lid if you want it completely closed and leak proof between every sip. The flip top works well and is easy to operate but becomes cumbersome if you are opening and closing it constantly.", "This is the 8th one I've bought! Somehow I keep 'lending' them to friends and they never come back! I've bought them for me, for my family, for my friends - for any coffee lover who wants to keep their coffee HOT & doesn't want to worry about the mug leaking. I seriously carry this around in my purse when commuting. It works so well, doesn't leak, easy to carry, fits in car cup holder, easy to clean and the clip is a great way to tote this around. I will continue to buy these from Amazon as well b/c as usual - they have the best prices!", "I honestly cannot recommend this product at all.As a side note: I have one of the original Contigo spill proof mugs with the auto seal lid, and it works great. If you want a good product, buy that one, but do not waste your money on this Contigo product.", "You want it to stay hot - buy this. You want it to stay cold - buy this. We had a contigo one from long ago which is nice but this ones design is better. We are very satisfied.", "I previously had this style mug for 6+ years and loved it. The paint did begin to chip but nothing broke on it. It did begin to collect liquid in the wall of it and started stinking so I purchased this new version. This new one does not seem to have the same quality of lid as the last one so only time will tell how it holds up.", "This is absolutely the very best travel mug I have ever purchased. This is a replacement mug for one that was gifted to me that my wife ruined by putting chai tea in it and leaving it for eight hours. The tea was still warm but I couldn't get the clove taste out of the cuo so I gave it to my wife and ordered me another one.", "It's clipped onto my bag or back pack. Same for everyday use. I can clip it onto my purse strap or even onto a coat pocket.TIPPINGThese mugs are easy to tip over. They are made for cup holders. I can't tell you how often I have caught a mug just about to go over. Here is a solution. Close the tab. If you shut the tab after a drink this thing can bounce across the kitchen tile and not open or spill.Just like everything, these cups have pros and cons.", "THIS IS A DANGEROUS PRODUCT THAT SHOULD NOT BE ON THE SHELVES. Please be warned, since there is no adequate warning on the mug or its instructions.", "I'd never use the carabiner clip, but it seems sturdy too. As far as keeping coffee hot, I'd give this a B.  Lukewarm, yes, but not hot. One definite minus is the rubbery plastic that lines the top of the mug on the inside. Not sure what the purpose is of that stuff, and I wonder if it will hold a coffee smell over time, or worse, allow water to leak into the mug walls, as another reviewer pointed out. Only time will tell. If this mug turns out to be a dud, I may try one of the handled Thermos models, though so far I haven't seen one that looks right to me."]}
reasoning:
1. Question focus: The user wants to know if the mug has a **taste of iron** when drinking from it.
2. Document scan: Reviewers frequently mention ease of cleaning, keeping coffee/tea hot, lid quality, portability, and durability.
3. Taste-related evidence: The reviews mention coffee or tea flavors lingering, clove taste issues, or rubbery plastic at the top, but **no mention of metallic or iron taste** is reported.
4. Material cues: The mug is compared with brands like Contigo and KleenKanteen, suggesting stainless steel or plastic components, but none of the reviews describe an iron-like aftertaste.
5. Conclusion: Based on user feedback, the mug does **not** have a taste of iron.
output: <answer>No. Based on customer reviews, the mug does not have an iron taste and users enjoy using it for coffee and tea.</answer>
""",
        'Product_Substitute_Identification': """
task: Product Substitute Identification
instruction: {instruction}
input: {input_data}
output: <answer>yes</answer> or <answer>no</answer>

Example:
task: Product Substitute IdentificationAnswer
instruction: Given a query and a product, identify if the product is somewhat relevant to the query. It fails to fulfill some aspects of the query but the product can be used as a functional substitute. Only output yes or no.
input: {"query": "obd2 scanner bluetooth ios iphone", "product": "OBD2 Scanner Bluetooth for iPhone, Bluetooth 4.0 OBDII Scan Tool for Android iOS (with Own Developed APP), Car Code Reader Diagnostic Tool to Clear Your Check Engine Light, Compatible with Torque Pro"}
reasoning:
1. Query intent: User is looking for an **OBD2 scanner** with **Bluetooth connectivity**, specifically for **iOS/iPhone**.
2. Product examination: The title clearly states "OBD2 Scanner Bluetooth for iPhone … for Android iOS," indicating iOS/iPhone compatibility.
3. Functionality match: Product functions as an OBDII scan tool, with Bluetooth 4.0, works with its own app and Torque Pro, and supports check engine light clearing.
4. Gap analysis: All aspects of the query (OBD2 + Bluetooth + iOS/iPhone) are fulfilled; it is not merely a substitute but an exact match.
5. Conclusion: Since the product fully satisfies the query, it is not a partial substitute — it is the correct item.
output: <answer>no</answer>
""",
        'Product_Relation_Prediction': """
task: Product Relation Prediction
instruction: {instruction}
input: {input_data}
output: <answer>A</answer> or <answer>B</answer> or ...

Example:
task: Product Relation Prediction
instruction: Given the title of two products, predict if the two products are similar, if the two products will be purchased or viewed together. Answer only from the options.
input: {"Product 1:": "Medieval Excalibur Knight Foam Padded Costume Prop Sword LARP", "Product 2:": "Oracle Fantasy Foam Sword LARP"}
options: ["A: Users who buy product 1 may also buy product 2.", "B: Users who view product 1 may also view product 2.", "C: The product 1 is similar with the product 2."]
reasoning:
1. Product 1: A medieval-themed **foam padded prop sword** (Excalibur style) for LARP or costume use.
2. Product 2: A fantasy-themed **foam sword** also intended for LARP.
3. Category match: Both are foam swords designed for similar activities (costume play / LARP).
4. Relationship type: While they are not identical, they appeal to the same audience. Users browsing one sword are likely to check out another variant.
5. Conclusion: They are more **viewed together** than directly bought as complementary items.
output: <answer>B</answer>
""",
        'Multiclass_Product_Classification': """
task: Multiclass Product Classification
instruction: {instruction}
input: {input_data}
output: <answer>A</answer> or <answer>B</answer> or ...

Example:
task: Multiclass Product Classification
instruction: Determine the relevance between the query and the product title provided, and select your response from one of the available options.（这句话不变）
input: {"query": "bathroom vanity tops 37 x 22", "product title": "ZEEK 37" x 22" Bathroom Vanity Top With Sink 3 Faucet Hole Vitreous China Ceramic White Countertop for 36 x 21 Vanity CT3708"}
options: ["A: The product is relevant to the query, and satisfies all the query specifications.", "B: The product is somewhat relevant. It fails to fulfill some aspects of the query but the product can be used as a functional substitute.", "C: The product does not fulfill the query, but could be used in combination with a product exactly matching the query.", "D: The product is irrelevant to the query."]
reasoning:
1. Query analysis: The user seeks a "bathroom vanity top" sized 37" x 22".
2. Product examination: Title specifies a "Bathroom Vanity Top With Sink" by ZEEK.
3. Dimension match: Listed dimensions are exactly "37" x 22"," matching the query size.
4. Category/type match: It is explicitly a bathroom vanity top, satisfying the product type.
5. Extra specs: 3 faucet holes and vitreous china material are additional features not required by the query and do not conflict.
6. Compatibility note: Designed for a 36" x 21" vanity base; the top dimension remains 37" x 22", which aligns with the query.
output: <answer>A: The product is relevant to the query, and satisfies all the query specifications.</answer>
""",
        'Sequential_Recommendation': """
task: Sequential Recommendation
instruction: {instruction}
input: {input_data}
output: <answer>A</answer> or <answer>B</answer> or ...

Example:
task: Sequential Recommendation
instruction: Given the products the user has purchased in history, rank the items in the listed options and output the item that the user is most likely to purchase next. Answer from one of the options.
input: ['1st: Glass Bottle w/Mist Sprayer 4oz. Home & Kitchen. Kitchen & Dining. Wyndmere Naturals.', '2nd: Meal Prep Haven 7 Piece Multi-Colored, Color Coded Portion Control Container Kit with Guide, Leak Proof, BPA Free, 21 Day Planner. Home & Kitchen. Kitchen...', '3rd: Fitpacker Meal Prep Containers - Portion Control Lunch Box (PACK OF 7). Home & Kitchen. Kitchen & Dining. Fitpacker.']
options: ['A: Dragon Touch Y88X Case,  Famavala Vegan Leather Case Cover For 7" Dragon Touch Y88X / Y88 / Q88 A13, IRULU eXpro Mini/X1a/X1s/Q8, 7" NeuTab...', 'B: Zeikos ZE-BLR Deluxe Dust Blower - Black. Electronics. Camera & Photo. Zeikos.', 'C: HP 435302-001 KB-0316 104 Key Black Silver PS2 Keyboard. Electronics. Computers & Accessories. HP.', 'D: Laundry Wash Bags - Reinforced Double Layered Mesh - Bonus Pink Bra Bag - Total 5 Pieces 2 Extra Large 2 Medium - Premium Quality...', 'E: Samsung WB150F Digital Camera Battery Charger (110/220v with Car & EU adapters) - Replacement Charger for Samsung SLB-10A, SLB-11A Battery. Electronics. Camera & Photo. Synergy...', 'F: Kootek 2 Pack Knee Strap Patella Tendon Brace Adjustable Neoprene Knee Pain Relief Patella Strap Band Support Brace Pads for Running, Jumpers Knee, Tennis, Basketball,...', 'G: Pioneer AVH-X3500BHS 2-DIN Multimedia DVD Receiver with 6.1 WVGA. Electronics. Car & Vehicle Electronics. Pioneer.', 'H: mDesign Over the Cabinet Kitchen Dish Towel Storage Hooks - Pack of 3, Assorted, Chrome. Home & Kitchen. Kitchen & Dining. mDesign.', 'I: SquareTrade 4-Year Camera & Camcorder Accidental Protection Plan ($50-74.99) - Basic. Electronics. Electronics Warranties. SquareTrade.', 'J: Ehdching Rectangular Silicone Loaf Toast Bread Pastry Cake Soap Mold Crafts Mould. Home & Kitchen. Kitchen & Dining. Ehdching.', 'K: Zinus Ultima Comfort Memory Foam 8 Inch Mattress, King. Home & Kitchen. Furniture. Zinus.', 'L: Cooper-Atkins FT24-0-3 Large Single Station Digital Timer, 24 Hour Digital with Volume Control, 24 Hours Unit Range. Home & Kitchen. Kitchen & Dining. Cooper.', 'M: SanDisk Ultra 32GB (5 Pack) USB 3.0 OTG Flash Drive with micro USB connector works with Android Mobile Devices - w/ (2) Everything But Stromboli...', 'N: 100 Cotton 5pcs Paris in Autumn Single Twin Size Comforter Set Eiffel Theme Bedding Linens. Home & Kitchen. Bedding. Bepoe HT.', 'O: Rigid Industries 40003 Trolling Motor Mount Light Kit. Sports & Outdoors. Sports & Fitness. Rigid Industries.', 'P: SPT WA-1140DE Dual-Hose 11,000-BTU Portable Air Conditioner with Remote Control. Home & Kitchen. Heating, Cooling & Air Quality. Sunpentown.', 'Q: New High Quality Sony CPA9C Cassette Adapter for iPod and iPhone. Electronics. Portable Audio & Video. Sony.', 'R: Honeywell L5200 Kit - LYNX Touch Wireless Security Alarm with (3) 5816WMWH Door/Window Transmitters, (1) 5834-4 Four-Button Wireless Keyfob and (1) 5800PIR-RES Wireless Motion Detector...', 'S: Yepal Unisex-Adult Waterproof Jewelry Roll Bag Hanging Jewelry Organizer Red. Home & Kitchen. Storage & Organization. Yepal.', "T: Small Geometric Screen with Doors, 38''W x 31''H, in Bronze. Home & Kitchen. Heating, Cooling & Air Quality. Plow & Hearth."]
reasoning:
1. User history shows a strong pattern around kitchen, dining, and home organization items**: spray bottles, meal prep containers, and portion control kits.
2. Options analysis: Most listed items are electronics, automotive, or unrelated accessories, which break the continuity of purchase behavior.
3. The most relevant category: Home & Kitchen remains the best match.
4. Within Home & Kitchen, option K (Zinus Ultima Comfort Memory Foam Mattress, King) stands out as a natural next-step home purchase, extending from smaller kitchen/dining utilities into broader home essentials.
5. Other kitchen-related items (H, J, L) are valid but less impactful compared to a large home-related purchase pattern shift, while K aligns with home lifestyle upgrading.
6. Conclusion: Option K is the most logical sequential recommendation.
output: <answer>K</answer>
"""
    }
    template = task_to_user_prompt.get(task, "")
    return template.replace("{instruction}", instruction).replace("{input_data}", input_data)


async def test_online_model(task):
    task_file = f"{task}_test.json"
    dataset_path = os.path.join(test_dataset_base_path, task_file)

    output_dir = f"res/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset path {dataset_path} does not exist. Skipping task {task}.")
        return

    try:
        logger.info(f"[{task}] Starting to load data...")
        with open(dataset_path, 'r') as file:
            data = json.load(file)
            data = data[:50]
            total_questions = len(data)
            logger.info(f"[{task}] Total {total_questions} items")

        all_results = []
        processed = 0

        for batch_start in range(0, total_questions, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_questions)
            batch_data = data[batch_start:batch_end]

            logger.info(f"[{task}] Processing batch {batch_start//BATCH_SIZE + 1}, data range: {batch_start+1}-{batch_end}")

            batch_tasks = []
            for idx, entry in enumerate(batch_data):
                real_idx = batch_start + idx
                batch_tasks.append(process_single_question_safe(task, real_idx, entry))

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            batch_success = 0
            batch_failed = 0
            for idx, result in enumerate(batch_results):
                real_idx = batch_start + idx
                if isinstance(result, Exception):
                    logger.error(f"[{task}] Question {real_idx + 1} failed: {result}")
                    batch_failed += 1
                    entry = batch_data[idx]
                    all_results.append({
                        "question_id": real_idx + 1,
                        "instruction": entry.get("instruction", ""),
                        "input": entry.get("input", ""),
                        "expected_output": entry.get("output", ""),
                        "output": f"ERROR: {str(result)}"
                    })
                else:
                    batch_success += 1
                    all_results.append(result)

            processed += len(batch_data)
            progress = (processed / total_questions) * 100
            logger.info(f"[{task}] Batch completed: Success {batch_success}, Failed {batch_failed} | Progress: {processed}/{total_questions} ({progress:.1f}%)")
            logger.info("-" * 50)

            if batch_end < total_questions:
                await asyncio.sleep(DELAY_BETWEEN_BATCHES)

        output_file = os.path.join(output_dir, f"{task}_results.json")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(all_results, outfile, indent=4, ensure_ascii=False)

        logger.info(f"[{task}] Test completed! Processed {len(all_results)} questions, results saved to {output_file}")
        return len(all_results)

    except Exception as e:
        logger.error(f"[{task}] Failed to read input file: {e}")
        return 0


async def process_single_question_safe(task, idx, entry):
    instruction = entry.get("instruction", "")
    input_data = entry.get("input", "")
    output = entry.get("output", "")

    logger.debug(f"[{task}] Starting to process item {idx + 1}...")

    try:
        response = await chat_ai_model_with_retry(task, instruction, input_data)
        logger.debug(f"[{task}] Item {idx + 1} processed successfully")
        return {
            "question_id": idx + 1,
            "instruction": instruction,
            "input": input_data,
            "expected_output": output,
            "output": response
        }
    except Exception as e:
        logger.error(f"[{task}] Item {idx + 1} failed: {e}")
        raise e


async def chat_ai_model_with_retry(task, instruction, input_data, max_retries=3):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": get_system_prompts_for_task(task)
                        },
                        {
                            "role": "user",
                            "content": get_user_prompt_for_task(task, instruction, input_data)
                        }
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + (attempt * 0.1)
                    logger.warning(f"[{task}] API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[{task}] API call final failure: {e}")
                    raise e


async def chat_ai_model(task, instruction, input_data):
    return await chat_ai_model_with_retry(task, instruction, input_data)

async def main():
    os.makedirs(f"res/{model_name}", exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Starting parallel testing for {len(tasks)} tasks...")
    logger.info(f"Task list: {tasks}")
    logger.info(f"Config: max_concurrent={MAX_CONCURRENT_REQUESTS}, batch_size={BATCH_SIZE}")
    logger.info("=" * 60)

    start_time = time.time()
    results = await asyncio.gather(*[test_online_model(task) for task in tasks], return_exceptions=True)
    end_time = time.time()

    total_processed = 0
    successful_tasks = 0
    failed_tasks = 0

    logger.info("\n" + "=" * 60)
    logger.info("Task execution results:")
    logger.info("-" * 60)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {tasks[i]} failed: {result}")
            failed_tasks += 1
        else:
            processed_count = result if isinstance(result, int) and result > 0 else 0
            logger.info(f"Task {tasks[i]}: Processed {processed_count} questions")
            total_processed += processed_count
            successful_tasks += 1

    logger.info("-" * 60)
    logger.info(f"Overall statistics:")
    logger.info(f"   Successful tasks: {successful_tasks}/{len(tasks)}")
    logger.info(f"   Failed tasks: {failed_tasks}/{len(tasks)}")
    logger.info(f"   Total processed: {total_processed} items")
    logger.info(f"   Total time: {end_time - start_time:.2f} seconds")

    if total_processed > 0:
        avg_time_per_question = (end_time - start_time) / total_processed
        logger.info(f"   Average time per item: {avg_time_per_question:.3f} seconds")

    logger.info("=" * 60)
    logger.info("All tasks completed!")

    return {
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks,
        'total_processed': total_processed,
        'total_time': end_time - start_time
    }

if __name__ == "__main__":
    asyncio.run(main())
