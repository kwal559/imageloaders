import sys,os,time,re,logging,time

ARG_SCHEMA = """
--prompt|app.py --prompt 'when i close my eyes, says the boy with an overactive imagintion, this is what i see'
--creativity|creativity levels 1(great), 3(pushing it), 5(its your time)|int
--num_ideas|x 3 models|int
"""
if len(sys.argv) == 1:
    print(ARG_SCHEMA.strip())
    sys.exit(0)

# ---(deferred slightly) ---
import torch, argparse
from transformers import pipeline

STORYTELLER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# EMBELLESHER MODELS
TEXT_MODELS = [
    'google/gemma-2-2b-it',
    'John6666/Llama-3.1-8B-Lexi-Uncensored-V2-nf4',
]

CONFIGS = {
    1: {'min_length': 70, 'max_length': 180, 'length_penalty': 1.2, 'repetition_penalty': 1.05, 'no_repeat_ngram_size': 3, 'num_beams': 3, 'temperature': 0.75, 'top_p': 0.85, 'top_k': 40, 'do_sample': True},
    3: {'min_length': 100, 'max_length': 250, 'length_penalty': 1.3, 'repetition_penalty': 1.1, 'no_repeat_ngram_size': 3, 'num_beams': 4, 'temperature': 0.9, 'top_p': 0.9, 'top_k': 50, 'do_sample': True},
    5: {'min_length': 150, 'max_length': 400, 'length_penalty': 1.4, 'repetition_penalty': 1.15, 'no_repeat_ngram_size': 4, 'num_beams': 5, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 60, 'do_sample': True}
}
timestamp=time.time()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def sanitize_filename_component(text, max_length=50):
    text = str(text)
    text = re.sub(r'[^\w\s-]', '', text).strip()
    text = re.sub(r'[-\s]+', '_', text)
    return text[:max_length]

def clean_text_for_prompt_file(text_block):
    if not text_block:
        return ""
    text_block = re.sub(r'<[^>]+>', '', text_block) # Remove HTML
    lines = [line.strip() for line in text_block.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"^(?:\d+\.?\s*)?(?:[\*\s]*Title[\*\s]*:)", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    cleaned = re.sub(r"^(?:[\*\s]*Prompt[\*\s]*:)", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    cleaned = re.sub(r"^(Here are the variations|Here are.*ideas|Here are.*prompts):", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    cleaned = re.sub(r"^(Okay, here.*|Sure, here.*):", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    if cleaned.lower().startswith("your curated collection"): cleaned = ""
    return cleaned.strip()

def generate_themed_ideas_from_llm(original_prompt, num_variations, llm_pipeline):
    logger.info(f"Everything will be saved, almost there..\nObfuscating Quigley Matrix\nRequesting {num_variations} themed ideas from Storytelling LLM for: '{original_prompt}'")
    variation_prompt_for_llm = (
        f"O, Master Storyteller, Weaver of Worlds! We come to you with a spark of an idea: '{original_prompt}'. We envision this not just as a simple image, but as a scene brimming with narrative potential, a frozen moment from an untold tale.\n\n"
        f"Our goal is to conjure {num_variations} truly distinct and captivating *visual scenarios* or *thematic concepts* born from '{original_prompt}'. Think of these as seeds for grand illustrations, each building and keeping reasonably within the original prompts subject.  We are not looking for full stories, just brief direction concepts to stimulate direction, a path to potentialy unlock quality outputs from an advanced image generation model. Avoid responses that are just memories or abstract reflections; aroma, taste, sound - avoid all that, we need visual prompts that only focus on tangible, visualizable scenes.\n\n"
        f"Unleash your famed direction!. The writing begins and ends with your help, we are counting on you!\n"
        f"Please present these {num_variations} as a hint towards a story waiting to be told, stemming from: '{original_prompt}'. Clearly number each concept."
    )
    messages = [{"role": "user", "content": variation_prompt_for_llm}]
    
    themed_ideas = []
    raw_llm_output_text = ""
    try:
        terminators = [
            llm_pipeline.tokenizer.eos_token_id,
            llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        response = llm_pipeline(
            messages,
            max_new_tokens=1200,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.12,
            top_k=60,
            do_sample=True,
            eos_token_id=terminators,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id
        )
        raw_llm_output_text = response[0]['generated_text'][-1]['content'].strip()
        logger.info(f"Populating Empyreal Entities\nRaw themed ideas output from LLM: \n{raw_llm_output_text}")

        matches = re.findall(r"^\s*\d+[\.\)]?\s*(.*)", raw_llm_output_text, re.MULTILINE)
        themed_ideas = [match.strip() for match in matches if match.strip() and len(match.strip().split()) > 5][:num_variations]

        if not themed_ideas and raw_llm_output_text:
            logger.warning("LLM did not return themed ideas in a numbered list format. Attempting to parse general lines.")
            potential_ideas = [line.strip() for line in raw_llm_output_text.split('\n') if line.strip() and len(line.strip().split()) > 5]
            themed_ideas = potential_ideas[:num_variations]
        
        if not themed_ideas:
             logger.warning("Could not parse distinct themed ideas from LLM. Using original prompt as sole idea.")
             themed_ideas = [original_prompt]
    except Exception as e: 
        logger.error(f"Error generating/parsing themed ideas: {e}. Raw output: '{raw_llm_output_text}'")
        themed_ideas = [original_prompt]

    logger.info(f"Successfully obtained {len(themed_ideas)} themed idea(s).\nRenewing Urban Combinatorics")
    return themed_ideas, raw_llm_output_text

def blend_and_refine_with_llm(original_prompt, embellished_drafts_text, llm_pipeline):
    logger.info("Requesting final blend and refinement from Storytelling LLM...")
    blend_prompt_for_llm = (
        f"Esteemed Storyteller, the stage is now yours for the grand finale! We began with the core concept: '{original_prompt}'. You then masterfully wove several initial thematic narrative threads. Subsequently, our team of specialized 'wordsmiths' embellished these threads, resulting in the collection below.\n\n"
        f"Now, we turn to your unparalleled editorial eye. From this wealth of embellished ideas, please distill the essence. We seek a small collection – perhaps 3 to 5 – of the most impactful, diverse, and visually compelling *premium* text-to-image prompts. Each should be a masterpiece of concise yet evocative language, ready to inspire breathtaking imagery. You must avoid overly abstract philosophical statements; focus on concrete, visualizable details and maintain the origial prompts general theme \n\n"
        f"Your signature touch of providing a clear, premium distinct prompt by separating each complete entry with '||' has proven incredibly effective. Please continue this format.\n\n"
        f"Survey these contributions. Identify the gold, discard the dross, and blend the finest elements with your creative fire. The final prompts should stand as paragons of clarity and quality visual details directly useful for image generation.\n\n"
        f"Here are the embellished concepts for your consideration:\n{embellished_drafts_text}\n\n"
        f"Your curated collection of titled, premium prompts (separated by ||):"
    )
    messages = [{"role": "user", "content": blend_prompt_for_llm}]
    
    blended_output_raw = ""
    try:
        terminators = [
            llm_pipeline.tokenizer.eos_token_id,
            llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        response = llm_pipeline(
            messages,
            max_new_tokens=2048,
            temperature=0.72,
            top_p=0.90,
            repetition_penalty=1.08,
            top_k=70,
            do_sample=True,
            eos_token_id=terminators,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id
        )
        blended_output_raw = response[0]['generated_text'][-1]['content'].strip()
        logger.info("Blending and refinement by LLM completed successfully.\nSeeding Architecture Simulation Parameters")
    except Exception as e:
        logger.error(f"Error blending prompts: {e}")
        blended_output_raw = "Error during blending. Raw drafts below:\n" + embellished_drafts_text
        
    return blended_output_raw

def main():
    parser = argparse.ArgumentParser(description="Enhance text-to-image prompts via LLM chaining.", add_help=False)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--creativity', type=int, default=1, choices=CONFIGS.keys())
    parser.add_argument('--num_ideas', type=int, default=3)
    
    cli_args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    prompt_snippet = sanitize_filename_component(cli_args.prompt)
    
    comprehensive_log_filename = os.path.join(output_dir, f"prompt_flow_log_{timestamp}_{prompt_snippet}.txt")
    all_prompts_filename = os.path.join(output_dir, f"all_prompts_collection_{timestamp}_{prompt_snippet}.txt")

    all_generated_prompts_for_collection_file = []

    with open(comprehensive_log_filename, 'w', encoding='utf-8') as log_f:
        log_f.write(f"COMPREHENSIVE PROMPT ENHANCEMENT LOG\nTimestamp: {timestamp}\n")
        log_f.write(f"Storytelling LLM: {STORYTELLER_MODEL}\n")
        log_f.write(f"Script Args: --prompt \"{cli_args.prompt}\" --creativity {cli_args.creativity} --num_ideas {cli_args.num_ideas}\n")
        log_f.write("="*50 + "\n\n1. ORIGINAL USER PROMPT:\n-------------------------\n" + cli_args.prompt + "\n\n")
        
        cleaned_original = clean_text_for_prompt_file(cli_args.prompt)
        if cleaned_original: all_generated_prompts_for_collection_file.append(cleaned_original)

        # === STAGE 1: IDEA GENERATION (Load -> Run -> Unload) ===
        logger.info(f"Loading main Storytelling LLM for idea generation: {STORYTELLER_MODEL}...")
        try:
            storyteller_pipeline = pipeline("text-generation", model=STORYTELLER_MODEL, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
            themed_ideas, raw_llm_themed_output = generate_themed_ideas_from_llm(cli_args.prompt, cli_args.num_ideas, storyteller_pipeline)
        except Exception as e:
            logger.error(f"FATAL: Failed during idea generation stage with {STORYTELLER_MODEL}. Error: {e}")
            sys.exit(1)
        finally:
            del storyteller_pipeline
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info("Unloaded Storytelling LLM from memory.")

        log_f.write(f"2. THEMED IDEAS FROM STORYTELLING LLM (Raw Output):\n--------------------------------------------------\n{raw_llm_themed_output}\n\n")
        log_f.write(f"   Parsed Themed Ideas ({len(themed_ideas)}):\n")
        for i, idea in enumerate(themed_ideas):
            log_f.write(f"   Idea {i+1}: {idea}\n")
            cleaned_idea = clean_text_for_prompt_file(idea)
            if cleaned_idea: all_generated_prompts_for_collection_file.append(cleaned_idea)
        log_f.write("\n")

        # === STAGE 2: EMBELLISHMENT (Loop, Load -> Run -> Unload) ===
        logger.info("Starting embellishment with smaller models...")
        embellished_drafts_list_raw = []
        selected_config_params = CONFIGS.get(cli_args.creativity, CONFIGS[1])
        log_f.write(f"3. EMBELLISHED DRAFTS FROM SMALLER MODELS:\n   (Using creativity level {cli_args.creativity} parameters: {selected_config_params})\n--------------------------------------------------\n")

        for model_name in TEXT_MODELS:
            log_f.write(f"\nProcessing with Model: {model_name}\n")
            logger.info(f"Loading embellisher model {model_name}...")
            try:
                embellisher_pipeline = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
                for i, idea_for_embellishment in enumerate(themed_ideas):
                    log_f.write(f"   Input Idea {i+1} for {model_name}: {idea_for_embellishment[:100]}...\n")
                    result = embellisher_pipeline(idea_for_embellishment, truncation=True, **selected_config_params)
                    full_text = result[0]['generated_text']
                    enhanced_text = full_text[len(idea_for_embellishment):].strip()
                    embellished_drafts_list_raw.append(enhanced_text)
                    log_f.write(f"   Output from {model_name} (Idea {i+1}):\n   {enhanced_text}\n\n")
                    cleaned_embellished = clean_text_for_prompt_file(enhanced_text)
                    if cleaned_embellished: all_generated_prompts_for_collection_file.append(cleaned_embellished)
            except Exception as e:
                error_msg = f"[Error with model {model_name}: {e}]"
                embellished_drafts_list_raw.append(error_msg)
                logger.error(error_msg)
                log_f.write(f"   {error_msg}\n")
            finally:
                del embellisher_pipeline
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                logger.info(f"Unloaded {model_name} from memory.")
        
        # === STAGE 3: FINAL BLENDING (Load -> Run -> Unload) ===
        unique_valid_raw_drafts_for_blending = list(dict.fromkeys(d for d in embellished_drafts_list_raw if not d.startswith("[Error:")))
        embellished_drafts_for_blending_input = "\n\n---\n\n".join(unique_valid_raw_drafts_for_blending) if unique_valid_raw_drafts_for_blending else "No valid drafts from text-to-text models."
        
        logger.info(f"Reloading main Storytelling LLM for final blending: {STORYTELLER_MODEL}...")
        try:
            storyteller_pipeline = pipeline("text-generation", model=STORYTELLER_MODEL, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
            final_blended_prompts_raw = blend_and_refine_with_llm(cli_args.prompt, embellished_drafts_for_blending_input, storyteller_pipeline)
        except Exception as e:
             logger.error(f"FATAL: Failed during final blending stage with {STORYTELLER_MODEL}. Error: {e}")
             final_blended_prompts_raw = "Error during blending. Raw drafts below:\n" + embellished_drafts_for_blending_input
        finally:
            del storyteller_pipeline
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info("Unloaded Storytelling LLM from memory for the final time.")


        log_f.write(f"4. FINAL BLENDED & REFINED PROMPTS FROM STORYTELLING LLM (Raw Output):\n----------------------------------------------------------\n{final_blended_prompts_raw}\n\n")
        
        final_prompts_for_main_output_file = []
        cleaned_final_blended_block = clean_text_for_prompt_file(final_blended_prompts_raw)
        if "||" in cleaned_final_blended_block:
            split_prompts = cleaned_final_blended_block.split('||')
            for p_block in split_prompts:
                p_block_final_cleaned = p_block.strip()
                if p_block_final_cleaned: final_prompts_for_main_output_file.append(p_block_final_cleaned)
        elif cleaned_final_blended_block:
            potential_prompts = [p.strip() for p in cleaned_final_blended_block.split('\n') if p.strip()]
            for p_pot in potential_prompts:
                if p_pot: final_prompts_for_main_output_file.append(p_pot)
        
        for final_p in final_prompts_for_main_output_file:
            if final_p and final_p not in all_generated_prompts_for_collection_file:
                 all_generated_prompts_for_collection_file.append(final_p)

        final_output_filename = os.path.join(output_dir, f"enhanced_prompts{timestamp}.txt")
        final_output_content = '\n||\n'.join(final_prompts_for_main_output_file).strip()
        with open(final_output_filename, 'w', encoding='utf-8') as f_out: f_out.write(final_output_content)
        
        logger.info(f"Saved final refined prompts to: {os.path.abspath(final_output_filename)}")
        log_f.write(f"5. FINAL OUTPUT (saved to {final_output_filename}):\n--------------------------------------------------\n{final_output_content}\n\nEND OF LOG\n")

    seen_in_collection = set()
    unique_all_prompts_collection = []
    for item in all_generated_prompts_for_collection_file:
        if item not in seen_in_collection:
            unique_all_prompts_collection.append(item)
            seen_in_collection.add(item)
    with open(all_prompts_filename, 'w', encoding='utf-8') as f_all: f_all.write('\n||\n'.join(unique_all_prompts_collection).strip())
    logger.info(f"Saved all collected prompts to: {os.path.abspath(all_prompts_filename)}")

    try:
        if sys.platform == "win32": os.startfile(os.path.abspath(final_output_filename))
        elif sys.platform == "darwin": os.system(f'open "{os.path.abspath(final_output_filename)}"')
        else: os.system(f'xdg-open "{os.path.abspath(final_output_filename)}"')
    except Exception as e: logger.warning(f"Could not automatically open main output file: {e}")

    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()