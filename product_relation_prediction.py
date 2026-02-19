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

        self.single_task = "Product_Relation_Prediction"

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
        "As an expert in E-commerce product relationship analysis, accurately analyze the titles of Product 1 and Product 2 to precisely select the option that clearly indicates their relationship using only title information, then explicitly respond strictly with the chosen option."
        )

    def get_few_shot_examples(self) -> str:
        Product_Relation_example = """
Product_Relation_Prediction Example 1
task:Product_Relation_Prediction
instruction:Given the title of two products, predict if the two products are similar, if the two products will be purchased or viewed together. Answer only from the options.
input:{"Product 1:": "Medieval Excalibur Knight Foam Padded Costume Prop Sword LARP", "Product 2:": "Oracle Fantasy Foam Sword LARP"}
option:["A: Users who buy product 1 may also buy product 2.", "B: Users who view product 1 may also view product 2.", "C: The product 1 is similar with the product 2."]

Reasoning:
1. Product 1: A medieval-themed **foam padded prop sword** (Excalibur style) for LARP or costume use.
2. Product 2: A fantasy-themed **foam sword** also intended for LARP.
3. Category match: Both are foam swords designed for similar activities (costume play / LARP).
4. Relationship type: While they are not identical, they appeal to the same audience. Users browsing one sword are likely to check out another variant.
5. Conclusion: They are more **viewed together** than directly bought as complementary items.
<answer>B</answer>

Product_Relation_Prediction Example 2
task:Product_Relation_Prediction
Given the title of two products, predict if the two products are similar, if the two products will be purchased or viewed together. Answer only from the options.
instruction:input:{"Product 1:": "SOTA LPC series III Lp Record Cleaning Machine with Waterproof Platter & 8oz. Crystal Fluid", "Product 2:": "VPI HW-16.5 Record Cleaning Machine with 8 Oz of solution and cleaning brush"}
option:["A: Users who buy product 1 may also buy product 2.", "B: Users who view product 1 may also view product 2.", "C: The product 1 is similar with the product 2."]

Reasoning:
1. Product 1: **SOTA LPC series III** record cleaning machine, includes waterproof platter and cleaning fluid.
2. Product 2: **VPI HW-16.5** record cleaning machine, includes solution and cleaning brush.
3. Both are high-end **vinyl record cleaning machines** designed for audiophiles.
4. Purchase vs. view: Since both products serve the exact same primary function (record cleaning machine), customers will usually compare them rather than buy both.
5. Conclusion: These products are **view alternatives**, meaning users viewing one will likely also view the other.
<answer>B</answer>
"""
        return Product_Relation_example

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
                        max_tokens=2048,
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
    print("ECInstruct Dataset Processor - Single Task (Product_Relation_Prediction) - Async")
    print("Process entire file (no train/test split), ordered by sequence number")
    print("Using DeepSeek OpenAI-Compatible API to generate answers")
    print("=" * 60)

    INPUT_FILE = "Product_Relation_Prediction.json"
    OUTPUT_FILE = "Product_Relation_Prediction_output.json"
    FAILED_IDS_FILE = "Product_Relation_Prediction_failed_ids.json"
    FAILED_ITEMS_FILE = "Product_Relation_Prediction_failed_items.json"
    MAX_CONCURRENT = 200
    SEMAPHORE_LIMIT = 100
    SAVE_INTERVAL = 1000

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
