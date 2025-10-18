#mypy_2tool.py
import os,gc,subprocess,psutil,sys,torch,random,winsound,requests,time,json
from PIL.PngImagePlugin import PngInfo

R, G, Y, B, M, C, W, X = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m', '\033[0m'
def flush(): gc.collect(); torch.cuda.empty_cache(); print(f"🧹✂️ {torch.cuda.memory_reserved()/1024**3:.1f}GB")

def memory(request=None):
    gi = subprocess.run(['nvidia-smi','--query-gpu=pstate,memory.used,temperature.gpu,utilization.gpu',
                         '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip().split(',')
    vu, ps, util, temp, ram = float(gi[1].strip().replace(" MiB", "")) / 1024, gi[0].strip(), gi[3].strip(), gi[2].strip(), psutil.virtual_memory()

    gpu_mem = int(gi[1].strip().replace(" MiB", "")) / 1024  
    system_mem = int(ram.used / 1024**3)    
    if request == "gpu":
        return gpu_mem
    elif request == "cpu":
        return system_mem
    else:
        print(f"{C}VRAM:{Y}{vu:.1f}{C}/24GB {Y}{ps} {C}{util} {temp}C | RAM:{Y}{ram.used/1024**3:.1f}{C}/64GB{X}")

def startup():
    print(f"{G}env:{Y}{os.environ.get('CONDA_DEFAULT_ENV', 'base')} {G}py:{Y}{sys.version.split()[0]} {G}torch:{Y}{torch.__version__} {G}cuda:{Y}{torch.version.cuda}{X}")
    procs = [(p.name(), p.memory_info().rss//1048576) for p in psutil.process_iter() if p.memory_info().rss > 50*1048576]
    top = sorted(procs, key=lambda x: x[1], reverse=True)[:5]
    colored = [f"{C}{name}:{Y}{mem}MB{X}" for name, mem in top]
    print(f"{B}Top processes: {', '.join(colored)}")

def check_versions():
    try: import triton; triton_v = triton.__version__
    except: triton_v = f"{R}Not installed{X}"
    try: import xformers; xf_v = xformers.__version__
    except: xf_v = f"{R}Not installed{X}"
    try: import flash_attn; fa_v = flash_attn.__version__
    except: fa_v = f"{R}Not installed{X}"
    print(f"{M}triton:{Y if 'Not' not in triton_v else ''}{triton_v} {M}xformers:{Y if 'Not' not in xf_v else ''}{xf_v} {M}flash-attn:{Y if 'Not' not in fa_v else ''}{fa_v}{X}")
    
def random_quote(txt_file=r"D:\audiovideo\random_processing_quotes.txt"): 
    with open(txt_file, 'r', encoding='utf-8') as file:
        print(random.choice(file.readlines()).strip())

def save_image(image, gen_params, folder=r"outputs\\", modtype="misc"):
    os.makedirs(folder, exist_ok=True)
    params_string = str(gen_params)
    metadata = PngInfo()
    metadata.add_text("parameters", params_string)
    filename = f"{folder}" + f"{modtype}" + f"img_{random.randint(1, 9999999)}.png"
    image.save(filename, pnginfo=metadata)
    os.startfile(filename)
    print(f"Saved: {filename}")

def play_sound(txt_audio=r"D:\audiovideo\voices\curtain\waiting.txt"):
    with open(txt_audio, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sound_path = random.choice(lines).strip()
    winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

def beep_done(): winsound.Beep(200, 80); winsound.Beep(300, 130)

def llm_batch_requests(model_path, requests_list, 
                      temperature=0.7, max_tokens=512,
                      kobold_exe=r"G:\llm\koboldcpp.exe", port=5001):
    """
    Process multiple LLM requests with the same model sequentially.
    requests_list = [(system_prompt, user_prompt, clear_context), ...]
    Returns dictionary with unique output names: {'output_llm_1': result1, 'output_llm_2': result2, ...}
    """
    print(f"{M}--- Batch LLM Processing ({len(requests_list)} requests) ---{X}")
    api_url = f"http://127.0.0.1:{port}"
    args = [kobold_exe, "--model", model_path, "--port", str(port), "--gpulayers", "-1",
            "--contextsize", "8192", "--flashattention", "--quiet", "--skiplauncher", "--singleinstance"]
    
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    kobold_process = subprocess.Popen(args, startupinfo=si)

    try:
        # Wait for server
        for i in range(90):
            try:
                response = requests.get(f"{api_url}/api/v1/model", timeout=1)
                if response.status_code == 200 and 'result' in response.json():
                    print(f"{G}Model loaded for {len(requests_list)} requests{X}")
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("LLM server failed to start.")

        results = {}
        for i, (system_prompt, user_prompt, clear_context) in enumerate(requests_list):
            request_num = i + 1
            output_name = f"output_llm_{request_num}"
            print(f"{C}--- Processing {output_name} ({request_num}/{len(requests_list)}) ---{X}")
            
            # Clear context if requested
            if clear_context:
                try:
                    requests.post(f"{api_url}/api/v1/abort", timeout=5)
                    time.sleep(1)
                    print(f"{Y}Context cleared for {output_name}{X}")
                except:
                    print(f"{Y}Context clear attempted for {output_name}{X}")

            # Build and send request
            full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            payload = {
                "prompt": full_prompt, 
                "max_tokens": max_tokens, 
                "temperature": temperature
            }
            
            print(f"{Y}Generating {output_name}...{X}")
            response = requests.post(f"{api_url}/v1/completions", json=payload, timeout=300)
            response.raise_for_status()
            result = response.json().get('choices', [{}])[0].get('text', '').strip()
            
            # Store with unique name
            results[output_name] = result
            print(f"{G}✓ {output_name} complete ({len(result)} chars){X}\n")

        return results

    except Exception as e:
        print(f"{R}Batch processing failed: {e}{X}")
        # Return partial results with error for remaining items
        error_results = results.copy() if 'results' in locals() else {}
        remaining = len(requests_list) - len(error_results)
        for i in range(remaining):
            error_results[f"output_llm_{len(error_results) + 1}"] = f"Error: {e}"
        return error_results
    finally:
        if kobold_process and kobold_process.poll() is None:
            print(f"{M}Shutting down LLM{X}")
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(kobold_process.pid)], 
                         capture_output=True, check=False)
# Usage examples:

# python
# from mypy_2tool import llm_batch_requests

# # Example 1: Basic usage with default temperature and max_tokens
# requests = [
#     ("You are a creative writer", "Describe a haunted mansion", True),
#     ("You are a technical expert", "Explain blockchain", True), 
#     ("You are a poet", "Write a haiku about space", True)
# ]

# results = llm_batch_requests(
#     model_path=r"G:\llm\vision\gemma327b_vllm_itq4.gguf",
#     requests_list=requests
# )

# # Access results by unique names
# print(results['output_llm_1'])  # First result
# print(results['output_llm_2'])  # Second result
# print(results['output_llm_3'])  # Third result

# # Example 2: Custom temperature and max_tokens
# results = llm_batch_requests(
#     model_path=r"G:\llm\vision\gemma327b_vllm_itq4.gguf",
#     requests_list=requests,
#     temperature=0.9,      # More creative
#     max_tokens=1024       # Longer responses
# )

# # Example 3: Mixed clear_context settings
# mixed_requests = [
#     ("You are a storyteller", "Begin a fantasy story", True),     # Clear context
#     ("Continue the story", "What happens next?", False),          # Don't clear context
#     ("You are a chef", "Give me a pizza recipe", True)            # Clear context again
# ]

# results = llm_batch_requests(
#     model_path=r"G:\llm\vision\gemma327b_vllm_itq4.gguf",
#     requests_list=mixed_requests,
#     temperature=0.8,
#     max_tokens=768
# )


def llm_request(model_path, system_prompt, user_prompt, 
                clear_context=True, max_tokens=512, temperature=0.7,
                kobold_exe=r"G:\llm\koboldcpp.exe", port=5001):
    """
    Flexible LLM request function - start model, send prompt, get response, shutdown.
    Perfect for importing and calling from other scripts.
    """
    print(f"{M}--- LLM Request ---{X}")
    api_url = f"http://127.0.0.1:{port}"
    args = [kobold_exe, "--model", model_path, "--port", str(port), "--gpulayers", "-1", 
            "--contextsize", "8192", "--flashattention", "--quiet", "--skiplauncher", "--singleinstance"]
    
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    kobold_process = subprocess.Popen(args, startupinfo=si)

    try:
        # Wait for server
        for i in range(90):
            try:
                response = requests.get(f"{api_url}/api/v1/model", timeout=1)
                if response.status_code == 200 and 'result' in response.json():
                    print(f"{G}Model loaded: {os.path.basename(model_path)}{X}")
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("LLM server failed to start.")

        # Clear context if requested
        if clear_context:
            try:
                requests.post(f"{api_url}/api/v1/abort", timeout=5)
                time.sleep(1)
                print(f"{C}Context cleared{X}")
            except:
                pass

        # Build the full prompt
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        
        # Send request
        payload = {
            "prompt": full_prompt, 
            "max_tokens": max_tokens, 
            "temperature": temperature
        }
        
        print(f"{Y}Generating...{X}")
        response = requests.post(f"{api_url}/v1/completions", json=payload, timeout=300)
        response.raise_for_status()
        result = response.json().get('choices', [{}])[0].get('text', '').strip()
        
        print(f"{G}Response ready ({len(result)} chars){X}")
        return result

    except Exception as e:
        print(f"{R}LLM request failed: {e}{X}")
        return f"Error: {e}"
    finally:
        # Always shutdown
        if kobold_process and kobold_process.poll() is None:
            print(f"{M}Shutting down LLM{X}")
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(kobold_process.pid)], 
                         capture_output=True, check=False)


# # Single request
# result = llm_request(
#     model_path=r"G:\llm\vision\gemma327b_vllm_itq4.gguf",
#     system_prompt="You are an Apex-tier Prompt Architect...", 
#     user_prompt="a cyberpunk samurai in neon rain",
#     clear_context=True
# )
# print(result)



## #explore dict
# examples="""dict/key
# prompt="casa de naranja"
# gen_params = {"num_inference_steps": 30,"width": 1536,"height": 640,"prompt": prompt}

# --#change value
# prompt="casa de azul"
# gen_params["prompt"] = prompt
# --#add value
# gen_params["max_cookies"] = 5

# --#return value and remove
# -gen_params.pop("max_cookies")      # removes and returns the value
# -del gen_params["max_cookies"]      # removes without returning
# -gen_params.pop("max_cookies", None) #if key might not exist, use pop

# --gen_params is just the dictionary itself.
#   **gen_params "unpacks" the dictionary into keyword arguments for a function call.
# -def make_cookie(prompt, width, height):
#     return f"{prompt} - {width}x{height}"
# make_cookie(**gen_params)
# -# same as: make_cookie(prompt=gen_params["prompt"], width=gen_params["width"], height=gen_params["height"])

# --use only wanted
# allowed = {k: gen_params[k] for k in ["prompt","width","height"]}
# make_cookie(**allowed)

# --Let the function accept arbitrary kwargs:
# def make_cookie(prompt, width, height, **kwargs):
#     # kwargs will catch extras like num_inference_steps
#     return f"{prompt} - {width}x{height}"


# ***Strict functions → don’t allow unknown kwargs → TypeError.
# ***Flexible functions → define **kwargs → silently swallow extras.
# """





# # mypy_2tool.py (Revised for Stateless, Controlled Generation)

# # ... (keep all your other helper functions: flush, memory, startup, etc.) ...

# def generate_prompt_suite(base_prompt,
#                           num_variations=2,
#                           kobold_exe=r"G:\llm\koboldcpp.exe",
#                           model_path=r"G:\llm\vision\gemma327b_vllm_itq4.gguf",
#                           port=5001):
#     """
#     Conducts a multi-step, STATELESS process with an LLM to generate
#     complete and properly constrained prompt suites for Stable Diffusion 3.
#     """
#     print(f"{M}--- Initializing LLM Creative Session ---{X}")
#     api_url = f"http://127.0.0.1:{port}" # Corrected loopback address
#     args = [kobold_exe, "--model", model_path, "--port", str(port), "--gpulayers", "-1", "--contextsize", "8192", "--flashattention", "--quiet", "--skiplauncher", "--singleinstance"]
    
#     si = subprocess.STARTUPINFO()
#     si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
#     kobold_process = subprocess.Popen(args, startupinfo=si)

#     try:
#         for i in range(90):
#             try:
#                 response = requests.get(f"{api_url}/api/v1/model", timeout=1)
#                 if response.status_code == 200 and 'result' in response.json():
#                     print(f"{G}LLM Server is ready.{X}")
#                     break
#             except requests.exceptions.RequestException:
#                 time.sleep(1)
#         else:
#             raise RuntimeError("LLM server failed to start.")

#         all_suites = []
#         current_base_prompt = base_prompt
#         for i in range(num_variations):
#             print(f"\n{C}--- Generating Prompt Suite {i+1}/{num_variations} ---{X}")
#             suite = {"base_prompt": current_base_prompt}

#             # STEP 1: Epic Prompt (for T5) - No length constraint here.
#             instruction_1 = f"You are an Apex-tier Prompt Architect. Your task is to transmute a user's core concept into a single, superlatively detailed text-to-image prompt. Your output MUST BE ONLY the prompt and nothing else.\nUser concept: '{current_base_prompt}'"
#             prompt_3 = _ask_llm(instruction_1, api_url, max_tokens=512)
#             suite['prompt_3'] = prompt_3
#             print(f"  {Y}✓ Generated Epic Prompt (for T5){X}")

#             # STEP 2: Tags (for CLIP) - WITH length constraint.
#             instruction_2 = f"You are a prompt analyst. Based on the following master prompt, distill its essence into a concise, comma-separated list of critical keywords. **CRITICAL CONSTRAINT: The output must be well under the 77 token limit.**\nMaster prompt: '{prompt_3}'"
#             prompt_tags = _ask_llm(instruction_2, api_url, max_tokens=100) # max_tokens is a fallback
#             suite['prompt'] = prompt_tags
#             print(f"  {Y}✓ Generated Tags (for CLIP){X}")

#             # STEP 3: Style (for OpenCLIP) - WITH length constraint.
#             instruction_3 = f"You are an art director. For the scene described in the master prompt, describe the artistic style (medium, lighting, color, artist influences) as a comma-separated list. **CRITICAL CONSTRAINT: The output must be well under the 77 token limit.**\nMaster prompt: '{prompt_3}'"
#             prompt_style = _ask_llm(instruction_3, api_url, max_tokens=100)
#             suite['prompt_2'] = prompt_style
#             print(f"  {Y}✓ Generated Style (for OpenCLIP){X}")
            
#             # STEP 4: Negative Prompt - WITH length constraint.
#             instruction_4 = f"You are a quality control specialist. Based on the master prompt, list common visual flaws or unwanted elements as a comma-separated list of negative keywords. **CRITICAL CONSTRAINT: Keep the list concise and effective.**\nMaster prompt: '{prompt_3}'"
#             negative_prompt = _ask_llm(instruction_4, api_url, max_tokens=100)
#             suite['negative_prompt'] = negative_prompt
#             print(f"  {Y}✓ Generated Negative Prompt{X}")
#             print(f" prompt - {prompt_tags}\n prompt2 - {prompt_style}\n prompt3 - {prompt_3}\n neg - {negative_prompt}")
#             all_suites.append(suite)

#             # STEP 5: Brainstorm a new scenario for the next loop.
#             if i < num_variations - 1:
#                 instruction_5 = f"You are a creative writer. Take the original core concept of '{base_prompt}' and imagine a completely different scenario or setting for it. Describe this new scene in a single, compelling sentence. Your output MUST be ONLY the sentence."
#                 current_base_prompt = _ask_llm(instruction_5, api_url, max_tokens=100)
#                 print(f"  {M}✓ Brainstormed new scenario: {current_base_prompt}{X}")

#         return all_suites

#     except Exception as e:
#         print(f"{R}An error occurred during prompt generation: {e}{X}")
#         return [{"prompt": base_prompt, "prompt_2": base_prompt, "prompt_3": base_prompt, "negative_prompt": ""}]
#     finally:
#         if kobold_process and kobold_process.poll() is None:
#             print(f"{M}--- Shutting down LLM Server ---{X}")
#             subprocess.run(["taskkill", "/F", "/T", "/PID", str(kobold_process.pid)], capture_output=True, check=False)

# def _ask_llm(instruction, api_url, max_tokens=512, temperature=0.7):
#     """Sends a single, stateless request to the LLM."""
#     payload = {"prompt": instruction, "max_tokens": max_tokens, "temperature": temperature}
#     response = requests.post(f"{api_url}/v1/completions", json=payload, timeout=180)
#     response.raise_for_status()
#     text = response.json().get('choices', [{}])[0].get('text', '').strip()
#     return text.replace('"', '').strip()







# def generate_prompt_suite(base_prompt,
#                           num_variations=2, # How many complete prompt sets to generate
#                           kobold_exe=r"G:\llm\koboldcpp.exe",
#                           model_path=r"G:\llm\vision\gemma312b_vllm_itq4.gguf",
#                           port=5001):
#     """
#     Conducts a multi-step conversation with an LLM to generate complete
#     prompt suites for Stable Diffusion 3, including variations.
#     """
    
#     # --- Start the LLM Server (if not already running) ---
#     # This logic assumes you want the function to manage the server.
#     # For a long session, you might start it once manually.
#     print(f"{M}--- Initializing LLM Creative Session ---{X}")
#     api_url = f"http://127.0.0.1:{port}"
#     args = [kobold_exe, "--model", model_path, "--port", str(port), "--gpulayers", "-1", "--contextsize", "8192", "--flashattention", "--quiet", "--skiplauncher", "--singleinstance"]
    
#     si = subprocess.STARTUPINFO()
#     si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
#     kobold_process = subprocess.Popen(args, startupinfo=si)

#     try:
#         # Wait for server to be ready
#         for i in range(90):
#             try:
#                 response = requests.get(f"{api_url}/api/v1/model", timeout=1)
#                 if response.status_code == 200 and 'result' in response.json():
#                     print(f"{G}LLM Server is ready.{X}")
#                     break
#             except requests.exceptions.RequestException:
#                 time.sleep(1)
#         else:
#             raise RuntimeError("LLM server failed to start.")

#         # --- The Conversation ---
#         all_suites = []
#         for i in range(num_variations):
#             print(f"\n{C}--- Generating Prompt Suite {i+1}/{num_variations} ---{X}")
#             conversation_history = ""
#             suite = {"base_prompt": base_prompt}

#             # INSTRUCTION TEMPLATE
#             # This is the core instruction set for the LLM's persona
#             persona = (
#                 "You are an Apex-tier Prompt Architect. You are a creative partner to a human artist. "
#                 "You will follow a multi-step process to build a complete prompt set for an advanced text-to-image model. "
#                 "Your responses must be concise, directly usable, and contain ONLY the requested information without any preamble or explanation."
#             )

#             # STEP 1: The Epic Prompt (for prompt_3)
#             instruction_1 = f"The user's core concept is: '{base_prompt}'. First, transmute this into a single, superlatively detailed, and artistically profound text-to-image prompt. This is the master blueprint."
#             prompt_3 = _ask_llm(persona, conversation_history, instruction_1, api_url)
#             suite['prompt_3'] = prompt_3
#             conversation_history += f"Human: {instruction_1}\nAI: {prompt_3}\n"
#             print(f"  {Y}✓ Generated Epic Prompt (for T5){X}")

#             # STEP 2: The Tags (for prompt)
#             instruction_2 = "Excellent. Now, based on the master blueprint you just created, distill its essence into a concise, comma-separated list of the most critical keywords and tags. Focus on subject, composition, and key objects."
#             prompt_tags = _ask_llm(persona, conversation_history, instruction_2, api_url, max_tokens=100)
#             suite['prompt'] = prompt_tags
#             conversation_history += f"Human: {instruction_2}\nAI: {prompt_tags}\n"
#             print(f"  {Y}✓ Generated Tags (for CLIP){X}")

#             # STEP 3: The Style (for prompt_2)
#             instruction_3 = "Perfect. For that same scene, describe the artistic style. Focus ONLY on medium, lighting, color palette, and artist influences. Format as a comma-separated list."
#             prompt_style = _ask_llm(persona, conversation_history, instruction_3, api_url, max_tokens=100)
#             suite['prompt_2'] = prompt_style
#             conversation_history += f"Human: {instruction_3}\nAI: {prompt_style}\n"
#             print(f"  {Y}✓ Generated Style (for OpenCLIP){X}")
            
#             # STEP 4: The Negative Prompt
#             instruction_4 = "Finally, what should be avoided to ensure the highest quality? List common visual flaws, unwanted elements, or conflicting concepts as a comma-separated list of negative keywords."
#             negative_prompt = _ask_llm(persona, conversation_history, instruction_4, api_url, max_tokens=100)
#             suite['negative_prompt'] = negative_prompt
#             print(f"  {Y}✓ Generated Negative Prompt{X}")
            
#             all_suites.append(suite)

#             # Reset for the next variation, asking for a new scenario
#             base_prompt = _ask_llm(persona, "", f"Now, take the original core concept of '{all_suites[0]['base_prompt']}' and imagine a completely different scenario or setting for it. Describe this new scene in one sentence.", api_url, max_tokens=100)
#             print(f"  {M}✓ Brainstormed new scenario: {base_prompt}{X}")

#         return all_suites

#     except Exception as e:
#         print(f"{R}An error occurred during prompt generation: {e}{X}")
#         return [{"prompt": base_prompt, "prompt_2": base_prompt, "prompt_3": base_prompt, "negative_prompt": ""}] # Fallback
#     finally:
#         # --- Shutdown the LLM Server ---
#         if kobold_process and kobold_process.poll() is None:
#             print(f"{M}--- Shutting down LLM Server ---{X}")
#             subprocess.run(["taskkill", "/F", "/T", "/PID", str(kobold_process.pid)], capture_output=True, check=False)

# def _ask_llm(persona, history, instruction, api_url, max_tokens=512, temperature=0.7):
#     """Helper function to send a single request to the LLM."""
#     full_prompt = f"{persona}\n\n{history}Human: {instruction}\nAI:"
#     payload = {
#         "prompt": full_prompt,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#     }
#     response = requests.post(f"{api_url}/v1/completions", json=payload, timeout=180)
#     response.raise_for_status()
#     data = response.json()
#     # Clean up the response
#     text = data.get('choices', [{}])[0].get('text', '').strip()
#     return text.replace('"', '').replace("AI:", "").strip()






# #update to use prompt_2 or however decided
# def enhance_prompt(prompt,
#                    profile='epic', #epic or concise
#                    kobold_exe=r"G:\llm\koboldcpp.exe", 
#                    model_path=r"G:\llm\vision\gemma312b_vllm_itq4.gguf",
#                    port=5001):

#     print(f"{M}   ...Enhancing prompt...{X}")
    
#     api_url = f"http://127.0.0.1:{port}"
#     args = [
#         kobold_exe, "--model", model_path, "--port", str(port),
#         "--gpulayers", "-1", "--contextsize", "4096", "--flashattention",
#         "--quiet", "--skiplauncher", "--singleinstance"
#     ]
    
#     si = subprocess.STARTUPINFO()
#     si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
#     kobold_process = subprocess.Popen(args, startupinfo=si)

#     try:
#         server_ready = False
#         for i in range(90): 
#             try:
#                 response = requests.get(f"{api_url}/api/v1/model", timeout=1)
#                 if response.status_code == 200 and response.json().get('result') != 'N/A':
#                     print(f"{G}...{X}")
#                     server_ready = True
#                     break
#             except requests.exceptions.RequestException:
#                 time.sleep(1)

#         if not server_ready:
#             print(f"{R}Error: LLM server didn't start.{X}")
#             return prompt

#         if profile == 'concise':
#             print(f"   ...{Y}Using concise profile{X}..")
#             llm_instructions = (
#                 f"You are an Apex-tier Prompt Architect. Your task is to transmute the user's core concept into a single, superlatively detailed text-to-image prompt. "
#                 f"Your output MUST BE ONLY the single, complete prompt and nothing else. "
#                 f"**CRITICAL CONSTRAINT: The final masterpiece must be a powerful, concise statement under 75 tokens.** "
#                 f"The user's core concept is: '{prompt}'.\n\n"
#                 f"Now, forge the ultimate, concise visual blueprint."
#             )
#             max_tokens = 100 
#         else: 
#             print(f"{C}   ...Using verbose LLM{X}..")
#             llm_instructions = (
#                 f"You are an Apex-tier Prompt Architect, a master of visual narrative. Your task is to transmute the user's core concept into a single, superlatively detailed, and artistically profound text-to-image prompt. "
#                 f"Your output MUST BE ONLY the single, complete prompt and nothing else. No preamble, titles, or commentary. "
#                 f"Focus exclusively on visually depictable elements: light, shadow, color, composition, perspective, artistic styles, subject expression, and texture. Exclude non-visuals like smells or inner thoughts. "
#                 f"The user's core concept is: '{prompt}'.\n\n"
#                 f"Now, forge the ultimate visual blueprint."
#             )
#             max_tokens = 512

#         payload = {"prompt": llm_instructions,"max_tokens": max_tokens,"temperature": 0.7,}
        
#         response = requests.post(f"{api_url}/v1/completions", json=payload, timeout=180)
#         response.raise_for_status()
#         data = response.json()
#         enhanced = data.get('choices', [{}])[0].get('text', '').strip().replace('"', '')

#         if enhanced:
#             print(f"{G}   ...complete!{X}")
#             return enhanced
#         else:
#             print(f"{Y}LLM returned empty. Using original prompt.{X}")
#             return prompt
#     except Exception as e:
#         print(f"{R}An error occurred during prompt enhancement: {e}{X}")
#         return prompt
#     finally:
#         if kobold_process and kobold_process.poll() is None:
#             subprocess.run(["taskkill", "/F", "/T", "/PID", str(kobold_process.pid)], capture_output=True, check=False)
