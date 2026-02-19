import json
import asyncio
import os
import time
from typing import List, Dict, Any, Optional
import logging
import sys
from openai import AsyncOpenAI
from dotenv import load_dotenv
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ECInstructProcessor:
    def __init__(self, api_key: str, max_concurrent: int = 100, semaphore_limit: int = 50):
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(semaphore_limit)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        self.single_task = "Sequential_Recommendation"

        self.processed_data: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()

        self.total_processed = 0
        self.success_count = 0
        self.error_count = 0

        self.failed_ids: List[Any] = []
        self.failed_items: List[Dict[str, Any]] = []

    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Loading dataset from {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"File {file_path} not found!")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {file_path}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return []

        if not isinstance(data, list):
            logger.error("Dataset should be a list of items")
            return []

        def seq_key(x):
            v = x.get('序号')
            try:
                return int(v)
            except:
                return float('inf')

        for i, item in enumerate(data):
            item['_orig_idx'] = i

        data.sort(key=lambda x: (seq_key(x), x['_orig_idx']))

        logger.info(f"Loaded {len(data)} total items; will process ALL items (no split filtering).")
        return data

    def get_system_prompt(self) -> str:
        return (
             "As an expert in sequential product recommendation, estimate the user's intent based on their purchase history, then predict which product they are most likely to purchase next from the given options, then respond only with the predicted product."
        )

    def get_few_shot_examples(self) -> str:
        Sequential_Recommendation_example = """
Sequential_Recommendation Example 1
task:Sequential_Recommendation
instruction:Given the products the user has purchased in history, rank the items in the listed options and output the item that the user is most likely to purchase next. Answer from one of the options.
input:['1st: Glass Bottle w/Mist Sprayer 4oz. Home & Kitchen. Kitchen & Dining. Wyndmere Naturals.', '2nd: Meal Prep Haven 7 Piece Multi-Colored, Color Coded Portion Control Container Kit with Guide, Leak Proof, BPA Free, 21 Day Planner. Home & Kitchen. Kitchen...', '3rd: Fitpacker Meal Prep Containers - Portion Control Lunch Box (PACK OF 7). Home & Kitchen. Kitchen & Dining. Fitpacker.']
option:['A: Dragon Touch Y88X Case,  Famavala Vegan Leather Case Cover For 7" Dragon Touch Y88X / Y88 / Q88 A13, IRULU eXpro Mini/X1a/X1s/Q8, 7" NeuTab...', 'B: Zeikos ZE-BLR Deluxe Dust Blower - Black. Electronics. Camera & Photo. Zeikos.', 'C: HP 435302-001 KB-0316 104 Key Black Silver PS2 Keyboard. Electronics. Computers & Accessories. HP.', 'D: Laundry Wash Bags - Reinforced Double Layered Mesh - Bonus Pink Bra Bag - Total 5 Pieces 2 Extra Large 2 Medium - Premium Quality...', 'E: Samsung WB150F Digital Camera Battery Charger (110/220v with Car & EU adapters) - Replacement Charger for Samsung SLB-10A, SLB-11A Battery. Electronics. Camera & Photo. Synergy...', 'F: Kootek 2 Pack Knee Strap Patella Tendon Brace Adjustable Neoprene Knee Pain Relief Patella Strap Band Support Brace Pads for Running, Jumpers Knee, Tennis, Basketball,...', 'G: Pioneer AVH-X3500BHS 2-DIN Multimedia DVD Receiver with 6.1 WVGA. Electronics. Car & Vehicle Electronics. Pioneer.', 'H: mDesign Over the Cabinet Kitchen Dish Towel Storage Hooks - Pack of 3, Assorted, Chrome. Home & Kitchen. Kitchen & Dining. mDesign.', 'I: SquareTrade 4-Year Camera & Camcorder Accidental Protection Plan ($50-74.99) - Basic. Electronics. Electronics Warranties. SquareTrade.', 'J: Ehdching Rectangular Silicone Loaf Toast Bread Pastry Cake Soap Mold Crafts Mould. Home & Kitchen. Kitchen & Dining. Ehdching.', 'K: Zinus Ultima Comfort Memory Foam 8 Inch Mattress, King. Home & Kitchen. Furniture. Zinus.', 'L: Cooper-Atkins FT24-0-3 Large Single Station Digital Timer, 24 Hour Digital with Volume Control, 24 Hours Unit Range. Home & Kitchen. Kitchen & Dining. Cooper.', 'M: SanDisk Ultra 32GB (5 Pack) USB 3.0 OTG Flash Drive with micro USB connector works with Android Mobile Devices - w/ (2) Everything But Stromboli...', 'N: 100 Cotton 5pcs Paris in Autumn Single Twin Size Comforter Set Eiffel Theme Bedding Linens. Home & Kitchen. Bedding. Bepoe HT.', 'O: Rigid Industries 40003 Trolling Motor Mount Light Kit. Sports & Outdoors. Sports & Fitness. Rigid Industries.', 'P: SPT WA-1140DE Dual-Hose 11,000-BTU Portable Air Conditioner with Remote Control. Home & Kitchen. Heating, Cooling & Air Quality. Sunpentown.', 'Q: New High Quality Sony CPA9C Cassette Adapter for iPod and iPhone. Electronics. Portable Audio & Video. Sony.', 'R: Honeywell L5200 Kit - LYNX Touch Wireless Security Alarm with (3) 5816WMWH Door/Window Transmitters, (1) 5834-4 Four-Button Wireless Keyfob and (1) 5800PIR-RES Wireless Motion Detector...', 'S: Yepal Unisex-Adult Waterproof Jewelry Roll Bag Hanging Jewelry Organizer Red. Home & Kitchen. Storage & Organization. Yepal.', "T: Small Geometric Screen with Doors, 38''W x 31''H, in Bronze. Home & Kitchen. Heating, Cooling & Air Quality. Plow & Hearth."]

Reasoning:
1. User history shows a strong pattern around **kitchen, dining, and home organization items**: spray bottles, meal prep containers, and portion control kits.
2. Options analysis: Most listed items are electronics, automotive, or unrelated accessories, which break the continuity of purchase behavior.
3. The most relevant category: **Home & Kitchen** remains the best match.
4. Within Home & Kitchen, option **K (Zinus Ultima Comfort Memory Foam Mattress, King)** stands out as a natural next-step home purchase, extending from smaller kitchen/dining utilities into broader home essentials.
5. Other kitchen-related items (H, J, L) are valid but less impactful compared to a large home-related purchase pattern shift, while K aligns with home lifestyle upgrading.
6. Conclusion: Option **K** is the most logical sequential recommendation.
<answer>K</answer>

Sequential_Recommendation Example 2
task:Sequential_Recommendation
instruction:Given the products the user has purchased in history, rank the items in the listed options and output the item that the user is most likely to purchase next. Answer from one of the options.
input:['1st: Tumbl Trak Gymnastics Folding Tumbling Panel Mat. Sports & Outdoors. Sports & Fitness. Tumbl Trak.', '2nd: Primos Old Crow Call. Sports & Outdoors. Sports & Fitness. Primos Hunting.', '3rd: Light my Fire Replacement Swedish FireSteel Fire Starter for FireKnife. Sports & Outdoors. Outdoor Recreation. Light my Fire.', "4th: US MIlitary M-12 holster. Sports & Outdoors. Sports & Fitness. US Gov't. contract.", '5th: Knight & Hale Cottontail Rabbit Distress Call. Sports & Outdoors. Sports & Fitness. Knight & Hale.', '6th: DUCK COMMANDER Duck Picker Call. Sports & Outdoors. Sports & Fitness. DUCK COMMANDER.', '7th: Bushnell Bear Grylls Edition BackTrack Original G2 GPS Personal Locator and Digital Compass, Orange/Black. Sports & Outdoors. Sports & Fitness. Bushnell.', '8th: Official US Military Olive Drab BDU Quick Release Pistol Belt Extender.', '9th: Dry Pak TPU Clear Multi-Purpose Waterproof Case with a Top Clip Hole and Bottom Corner D-Rings. Sports & Outdoors. Sports & Fitness. Dry Pack.', '10th: Smith and Wesson SWPENMP2BK 5.8in Aircraft Aluminum Refillable Tactical Screw Cap Pen for Outdoor Survival Camping and Everyday Carry. Sports & Outdoors. Fan Shop. Smith...', '11th: US Army MOLLE II Pistolman Set, ACU. Sports & Outdoors. Sports & Fitness. Specialty Defense Systems.', '12th: ALPS OutdoorZ Big Bear Hunting Day Pack. Sports & Outdoors. Sports & Fitness. ALPS OutdoorZ.', '13th: DRY PAK DP-512 VHF Radio Case (5 Inch x 12 Inch). Sports & Outdoors. Sports & Fitness. Kwik Tek.', '14th: Reyes Industries, inc. US Military Enhanced Tactical Load Bearing Vest. Sports & Outdoors. Sports & Fitness. Reyes Industries, inc.', '15th: Crosman 1322C .22Cal Variable Pump. Sports & Outdoors. Sports & Fitness. Crosman.', '16th: M7 Shoulder Holster Black Leather. Sports & Outdoors. Sports & Fitness. World War Supply.', "17th: Columbia Sportswear Girl's Bugaboo Pants. Sports & Outdoors. Outdoor Recreation. Columbia.", '18th: Ka-Bar Fighting Knife with Kraton Handle Edge, Grey. Sports & Outdoors. Sports & Fitness. Ka-Bar.', '19th: KA5011-BRK Fighting Knife. Sports & Outdoors. Sports & Fitness. Ka-Bar.', '20th: Multicam Hydration Tactical Pack Drink Tube Cover Sleeve. Sports & Outdoors. Outdoor Recreation. Hydration Tube Covers.', '21st: CONDOR Tactical Sidekick Pouch - Black. Sports & Outdoors. Sports & Fitness. CONDOR.', '22nd: LEE PRECISION 90254 Classic Loader, 9mm Luger. Sports & Outdoors. Sports & Fitness. LEE PRECISION.', '23rd: Brunton TruArc 20 Compass. Sports & Outdoors. Sports & Fitness. Brunton.', '24th: Victorinox Swiss Army EvoGrip Army Knife. Sports & Outdoors. Outdoor Recreation. Victorinox.', '25th: Hydration Bladder 2 Liter, 1.5 & 3 Liter Water Bladder, Leak Proof Water Reservoir, Tasteless BPA-Free Military Water Storage Bladder Bag Hiking Climbing Cycling +...', '26th: Arbogast Triple Threat 3 - Pack - Legendary. Sports & Outdoors. Sports & Fitness. Arbogast.', '27th: Eagle Claw Practice Plugs 2 Pieces. Sports & Outdoors. Sports & Fitness. Eagle Claw.', '28th: ALPS Mountaineering MicroFiber Camp Pillow. Sports & Outdoors. Outdoor Recreation. ALPS Mountaineering.', '29th: Mens Tactical Gear Molle Hydration Ready Sling Shoulder Backpack Daypack Bag. Sports & Outdoors. Sports & Fitness. NPUSA.', '30th: Barlii Waterproof Dry Bag Sports Backpack 6L/10L/20L DrySak - 2 Shoulder Straps, Non-Toxic Roll Top Sack Outdoor Essentials for Kayak, SUP, Camping, Hiking, Cruise, Vacation,...', '31st: South Bend Practice Plugs. Sports & Outdoors. Sports & Fitness. South Bend.', "32nd: Zebco/Quantum 33L602M, 10C, NS4 Quantum, 33 Ladies Spincast Combo, 6' 2Piece Rod, 3.6: 1 Gear Ratio, Ambidextrous. Sports & Outdoors. Sports & Fitness. Zebco.", '33rd: EcoVessel BIGFOOT Triple Insulated Stainless Steel Water Bottle with Tea - Fruit and Ice Infuser. Sports & Outdoors. Sports & Fitness. EcoVessel.', '34th: Kwik Tek Dry PAK DPC-69 Multi-Purpose Case. Sports & Outdoors. Sports & Fitness. Dry Pack.', '35th: Piscifun Aluminum Fishing Pliers Braid Cutters Split Ring Pliers Hook Remover Fish Holder with Sheath and Lanyard. Sports & Outdoors. Sports & Fitness. Piscifun.', '36th: Gerber StrongArm 420 High Carbon Stainless Steel Fixed Blade Survival Tactical Knife with Molle Compatible Multi-Mount Sheath - Serrated Edge - Coyote Brown (30-001059). Sports...', '37th: JSHANMEI Split Rings Stainless Steel Fishing Tackle Ring Chain Connector High Strength Split Ring. Sports & Outdoors. Sports & Fitness. JSHANMEI.', '38th: MIRA Stainless Steel Vacuum Insulated Water Bottle | Leak-Proof Double Walled Cola Shape Bottle | Keeps Drinks Cold for 24 Hours & Hot for 12...', '39th: Proguard Skate Lace Hook Tightener Carded. Sports & Outdoors. Sports & Fitness. Pro Guard.', '40th: Mountainsmith Tour Lumbar Pack. Sports & Outdoors. Outdoor Recreation. Mountainsmith.', '41st: CamelBak Rain Cover. Sports & Outdoors. Outdoor Recreation. CamelBak.']
option:['A: LaCie Big Disk Hi-Speed Compact 1TB USB2.0 Hard Drive (300966U). Electronics. Computers & Accessories. LaCie.', 'B: Melange New Italian Villa Porcelain 20-Piece Place Setting, White, Service for 4. Home & Kitchen. Home Decor. Melange.', 'C: Outdoor Products Insulated Gel Reservoir, 2-Liter Storage. Sports & Outdoors. Outdoor Recreation. Outdoor Products.', 'D: Honeywell HWF101AB Replacement Filter for Filtration System HWB101 series. Home & Kitchen. Kitchen & Dining. Honeywell.', 'E: Tera [Upgraded Version] 18/8 Stainless Steel Manual Egg Whisk Hand Egg Mixer Eggbeater Blender Stirring Beater Cream Frother Flour Stirrer for Family Restaurant Kitchen. Home...', 'F: Proxxon 38704 Heavy Duty Transformer NG 5/E. Electronics. Accessories & Supplies. Proxxon.', 'G: (12) One Dozen Hand Blown Glass Pickle Christmas Tree Ornaments for Good Luck Trim-A-Tree Stocking Stuffer or Gift Giving. Home & Kitchen. Home Decor. Fun...', 'H: Imax 73327-3 Addison Ceramic Canisters - Tea Coffee Sugar Canisters, Vintage Inspired, Canister Storage Vault. Handcrafted Kitchenware. Home & Kitchen. Kitchen & Dining. Imax.', 'I: Mountainsmith Strappette Shoulder Straps. Sports & Outdoors. Outdoor Recreation. Mountainsmith.', 'J: Christy Renaissance 25 by 42 Large Rug, Alabaster. Home & Kitchen. Home Decor. Christy.', 'K: 1911 Compact Officer Grips "Cobra Skeleton Grips" Brushed. Sports & Outdoors. Sports & Fitness. Valkyrie Dynamics.', 'L: BCP 8pcs ITW MOLLE Web Dominators with Elastic String(Army Green,Black). Sports & Outdoors. Sports & Fitness. BCP.', 'M: Mission Enduracool Lockdown Reversible Cooling Headband. Sports & Outdoors. Sports & Fitness. MISSION.', 'N: Sweet Tooth Ceramic Cookie Jar. Home & Kitchen. Kitchen & Dining. Streamline.', 'O: Tripp Lite 7 Outlet Surge Protector Power Strip, 6ft Cord, Right-Angle Plug, Tel/Modem/Coax Protection, & $100,000 INSURANCE (HT706TSAT). Electronics. Accessories & Supplies. Tripp Lite.', 'P: Oumers Large Insulated Lunch Tote Bag Cooler Box - Red. Home & Kitchen. Kitchen & Dining.', 'Q: 6 Replacement Triangle Pads Compatible with Shark Euro Pro Pocket Steam Mop. Home & Kitchen. Vacuums & Floor Care. Astar.', 'R: Nikon COOLPIX L830 16 MP CMOS Digital Camera with 34x Zoom NIKKOR Lens and Full 1080p HD Video (Black) (Discontinued by Manufacturer). Electronics. Camera &...', 'S: Sigg Swiss Emblem Water Bottle. Sports & Outdoors. Sports & Fitness. Sigg.', 'T: Spy Tec K1S Dual Car Dash Camera | Front and Rear 1080p Remote Lens Cameras |. Electronics. Car & Vehicle Electronics. Koonlung.']

Reasoning:
1. User's purchase history is heavily focused on **Sports & Outdoors**, particularly **tactical gear, survival tools, hydration packs, knives, compasses, and outdoor accessories**.
2. Options analysis: Most choices are unrelated categories like electronics (A, F, O, T), kitchen/home decor (B, D, E, G, H, J, N, P, Q), which deviate from the strong outdoor/survival focus.
3. Narrowing down: The closest matching options are **C (Insulated Gel Reservoir)**, **I (Mountainsmith Strappette Shoulder Straps)**, **K (1911 Pistol Grips)**, **L (MOLLE web dominators)**, **M (Cooling Headband)**, and **S (Sigg Water Bottle)**.
4. Best alignment: Given the user already bought **tactical packs, hydration bladders, MOLLE gear, and Mountainsmith outdoor items**, option **I (Mountainsmith Strappette Shoulder Straps)** directly complements previously purchased gear, especially the Mountainsmith Tour Lumbar Pack (#40).
5. Conclusion: Option **I** is the most logical sequential recommendation.
<answer>I</answer>
"""
        return Sequential_Recommendation_example

    def get_user_prompt(self, instruction: str, input_data: Any, options: Optional[Any] = None) -> str:
        few_shot_str = self.get_few_shot_examples() or ""

        try:
            if isinstance(input_data, (dict, list)):
                input_content = json.dumps(input_data, ensure_ascii=False)
            else:
                input_content = str(input_data)
        except Exception:
            input_content = str(input_data)

        options_line = ""
        if options is not None:
            try:
                if isinstance(options, (list, tuple, dict)):
                    options_line = f"Option: {json.dumps(options, ensure_ascii=False)}\n"
                else:
                    options_line = f"Option: {str(options)}\n"
            except Exception:
                options_line = f"Option: {str(options)}\n"

        user_prompt = (
            f"Task: {self.single_task}\n"
            f"Instruction: {instruction}\n"
            f"Input: {input_content}\n"
            f"{options_line}"
        )

        if few_shot_str.strip():
            user_prompt += f"\nFew-Shot Examples:\n{few_shot_str.strip()}\n"

        user_prompt += (
            "\nPlease provide your reasoning step by step, then give your final answer. "
            "Do not use markdown formatting (no #, *, etc.). "
            "Put your final answer in <answer></answer> tags."
        )

        return user_prompt

    async def call_deepseek_api(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=2500,
                        temperature=0.7
                    )
                    content = response.choices[0].message.content
                    return content.strip() if content else ""
                except Exception as e:
                    msg = str(e)
                    if "rate_limit" in msg.lower() or "429" in msg:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"API call error (attempt {attempt + 1}): {msg}")
                        if attempt == max_retries - 1:
                            return ""
                        await asyncio.sleep(1)
            return ""

    def extract_final_answer(self, ai_response: str) -> str:
        if not ai_response.strip():
            return ""
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', ai_response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        answer_patterns = ["Answer:", "Final Answer:", "答案：", "最终答案：", "结论：", "Conclusion:"]
        for pattern in answer_patterns:
            if pattern in ai_response:
                answer_part = ai_response.split(pattern)[-1].strip()
                answer = answer_part.split('\n')[0].strip()
                if answer:
                    return answer
        lines = ai_response.strip().split('\n')
        return lines[-1].strip() if lines else ""

    def format_conversation(self, item: Dict[str, Any], ai_response: str) -> Dict[str, Any]:
        final_answer = self.extract_final_answer(ai_response) if ai_response.strip() else ""
        formatted_data = {
            "序号": item.get('序号'),
            "split": item.get('split'),
            "task": item.get('task', self.single_task),
            "setting": item.get('setting', ''),
            "instruction": item.get('instruction'),
            "input": item.get('input'),
            "options": item.get('options'),
            "output": item.get('output'),
            "adjust_output": final_answer,
            "ai_reasoning": ai_response
        }
        return formatted_data

    async def process_single_item(self, item: Dict[str, Any], index: int, total: int) -> Optional[Dict[str, Any]]:
        seq = item.get('序号', f"idx_{index+1}")
        try:
            system_prompt = self.get_system_prompt()
            user_prompt = self.get_user_prompt(
                item.get('instruction', ''),
                item.get('input', ''),
                item.get('options')
            )

            if index < 10:
                logger.info(f"\n===== USER PROMPT [{index+1}/{total}] seq={seq} =====\n{user_prompt}\n===== END USER PROMPT =====")

            try:
                with open("debug_user_prompts.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n\n===== USER PROMPT [{index+1}/{total}] seq={seq} =====\n{user_prompt}\n===== END USER PROMPT =====\n")
            except Exception as _e:
                logger.warning(f"Failed to write debug_user_prompts.txt: {_e}")

            ai_response = await self.call_deepseek_api(system_prompt, user_prompt)

            if ai_response:
                formatted_data = self.format_conversation(item, ai_response)
                async with self.lock:
                    self.processed_data.append(formatted_data)
                    self.success_count += 1
                logger.info(f"Processed item {index + 1}/{total} seq={seq}")
                return formatted_data
            else:
                async with self.lock:
                    self.error_count += 1
                    self.failed_ids.append(seq)
                    self.failed_items.append(item)
                logger.warning(f"Failed to generate response for item {index + 1}/{total} seq={seq}")
                return None

        except Exception as e:
            async with self.lock:
                self.error_count += 1
                self.failed_ids.append(seq)
                self.failed_items.append(item)
            logger.error(f"Error processing item {index + 1}/{total} seq={seq}: {str(e)}")
            return None
        finally:
            async with self.lock:
                self.total_processed += 1

    async def process_batch_async(self, items: List[Dict[str, Any]]) -> None:
        total_items = len(items)

        tasks = [
            self.process_single_item(item, i, total_items)
            for i, item in enumerate(items)
        ]

        batch_size = self.max_concurrent
        completed_count = 0

        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for j, result in enumerate(results):
                task_index = i + j
                completed_count += 1

                if isinstance(result, Exception):
                    logger.error(f"Task {task_index + 1} failed with exception: {str(result)}")
                elif result is None:
                    logger.debug(f"Task {task_index + 1} processing returned None")

                if completed_count % 10 == 0 or completed_count == total_items:
                    logger.info(f"Progress: {completed_count}/{total_items} items completed "
                                f"(Success: {self.success_count}, Errors: {self.error_count})")

    async def save_incremental(self, output_file: str):
        async with self.lock:
            if not self.processed_data:
                return

            try:
                existing_data = []
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    except (json.JSONDecodeError, Exception):
                        logger.warning(f"Could not read existing file {output_file}, will overwrite")
                        existing_data = []

                existing_keys = set()
                for item in existing_data:
                    key = f"{item.get('task', '')}|{item.get('instruction', '')}|{str(item.get('input', ''))}"
                    existing_keys.add(key)

                new_data = []
                duplicates_count = 0
                for item in self.processed_data:
                    key = f"{item.get('task', '')}|{item.get('instruction', '')}|{str(item.get('input', ''))}"
                    if key not in existing_keys:
                        new_data.append(item)
                        existing_keys.add(key)
                    else:
                        duplicates_count += 1

                if duplicates_count > 0:
                    logger.warning(f"Found {duplicates_count} duplicate items, skipped")

                all_data = existing_data + new_data

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)

                logger.info(f"Saved {len(new_data)} new items to {output_file} "
                            f"(Total: {len(all_data)} items, Skipped duplicates: {duplicates_count})")

                self.processed_data.clear()

            except Exception as e:
                logger.error(f"Error saving incremental data: {str(e)}")

    def save_failed_reports(self, failed_ids_file: str, failed_items_file: str):
        try:
            uniq_ids = list({x for x in self.failed_ids})
            def id_key(v):
                try:
                    return (0, int(v))
                except:
                    return (1, str(v))
            uniq_ids.sort(key=id_key)

            with open(failed_ids_file, 'w', encoding='utf-8') as f:
                json.dump(uniq_ids, f, ensure_ascii=False, indent=2)
            logger.info(f"Failed IDs saved to {failed_ids_file} (count={len(uniq_ids)})")

            with open(failed_items_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_items, f, ensure_ascii=False, indent=2)
            logger.info(f"Failed items saved to {failed_items_file} (count={len(self.failed_items)})")

        except Exception as e:
            logger.error(f"Error saving failed reports: {str(e)}")

    def create_backup(self, output_file: str):
        if os.path.exists(output_file):
            backup_file = output_file.replace('.json', f'_backup_{int(time.time())}.json')
            try:
                import shutil
                shutil.copy2(output_file, backup_file)
                logger.info(f"Created backup: {backup_file}")
            except Exception as e:
                logger.warning(f"Could not create backup: {str(e)}")

    async def process_dataset(self, input_file: str, output_file: str, failed_ids_file: str, failed_items_file: str, save_interval: int = 50):
        start_time = time.time()

        data = self.load_dataset(input_file)
        if not data:
            logger.warning("No data to process")
            self.save_failed_reports(failed_ids_file, failed_items_file)
            return

        self.create_backup(output_file)

        logger.info(f"Starting to process {len(data)} items (single task: {self.single_task})")
        logger.info(f"Configuration: max_concurrent={self.max_concurrent}, save_interval={save_interval}")

        for i in range(0, len(data), save_interval):
            chunk = data[i:i + save_interval]
            chunk_start_time = time.time()

            logger.info(f"Processing chunk {i//save_interval + 1}/{(len(data) + save_interval - 1)//save_interval} "
                        f"({len(chunk)} items)")

            await self.process_batch_async(chunk)
            await self.save_incremental(output_file)

            self.save_failed_reports(failed_ids_file, failed_items_file)

            chunk_time = time.time() - chunk_start_time
            logger.info(f"Chunk completed in {chunk_time:.2f} seconds")

        total_time = time.time() - start_time

        logger.info("Dataset processing completed!")
        logger.info(f"Statistics:")
        logger.info(f"   Total items processed: {self.total_processed}")
        logger.info(f"   Successful: {self.success_count}")
        logger.info(f"   Errors: {self.error_count}")
        logger.info(f"   Success rate: {(self.success_count/max(self.total_processed, 1)*100):.1f}%")
        logger.info(f"   Total time: {total_time:.2f} seconds")
        logger.info(f"   Average time per item: {(total_time/max(self.total_processed, 1)):.2f} seconds")

        self.validate_output_file(output_file)

    def validate_output_file(self, output_file: str):
        try:
            if not os.path.exists(output_file):
                logger.error(f"Output file {output_file} does not exist!")
                return

            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error("Output file should contain a list of data items")
                return

            unique_data = []
            seen_keys = set()
            duplicates_removed = 0

            for item in data:
                key = f"{item.get('task', '')}|{item.get('instruction', '')}|{str(item.get('input', ''))}"
                if key not in seen_keys:
                    unique_data.append(item)
                    seen_keys.add(key)
                else:
                    duplicates_removed += 1

            if duplicates_removed > 0:
                logger.warning(f"Found {duplicates_removed} duplicate items in final file, removing...")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Removed {duplicates_removed} duplicates from final file")

            logger.info(f"Output file validation passed")
            logger.info(f"Final file: {output_file}")
            logger.info(f"Contains {len(unique_data)} unique data items")
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate items")

        except Exception as e:
            logger.error(f"Error validating output file: {str(e)}")

def get_api_key():
    load_dotenv()
    api_key = os.getenv('DEEPSEEK_API_KEY2')
    if api_key:
        return api_key
    return None

async def main():
    print("=" * 60)
    print("ECInstruct Dataset Processor - Single Task (Sequential_Recommendation) - Async")
    print("Process entire file (no train/test split), ordered by sequence number")
    print("Using DeepSeek OpenAI-Compatible API to generate answers")
    print("=" * 60)

    INPUT_FILE = "Sequential_Recommendation.json"
    OUTPUT_FILE = "Sequential_Recommendation_output.json"
    FAILED_IDS_FILE = "Sequential_Recommendation_failed_ids.json"
    FAILED_ITEMS_FILE = "Sequential_Recommendation_failed_items.json"
    MAX_CONCURRENT = 200
    SEMAPHORE_LIMIT = 100
    SAVE_INTERVAL = 400

    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file {INPUT_FILE} not found!")
        return

    api_key = get_api_key()
    if not api_key:
        logger.error(f"DeepSeek API key is required! Please set DEEPSEEK_API_KEY in your .env")
        return

    logger.info(f"API key loaded (length: {len(api_key)})")

    logger.info(f"Input file: {INPUT_FILE}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Failed ids file: {FAILED_IDS_FILE}")
    logger.info(f"Failed items file: {FAILED_ITEMS_FILE}")
    logger.info(f"Max concurrent: {MAX_CONCURRENT}")
    logger.info(f"Semaphore limit: {SEMAPHORE_LIMIT}")
    logger.info(f"Save interval: {SAVE_INTERVAL}")

    try:
        processor = ECInstructProcessor(
            api_key=api_key,
            max_concurrent=MAX_CONCURRENT,
            semaphore_limit=SEMAPHORE_LIMIT
        )
    except Exception as e:
        logger.error(f"Failed to create processor: {str(e)}")
        return

    try:
        await processor.process_dataset(
            INPUT_FILE,
            OUTPUT_FILE,
            FAILED_IDS_FILE,
            FAILED_ITEMS_FILE,
            save_interval=SAVE_INTERVAL
        )
        logger.info("All done! Your dataset is ready.")

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        processor.save_failed_reports(FAILED_IDS_FILE, FAILED_ITEMS_FILE)
        logger.info("Partial results may have been saved")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        processor.save_failed_reports(FAILED_IDS_FILE, FAILED_ITEMS_FILE)
        logger.info("Check the logs above for more details")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            pass
