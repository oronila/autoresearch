import os
import time
import json
import subprocess
from openai import OpenAI

# Ensure API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Please set your OPENAI_API_KEY environment variable.")
    print("Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)

client = OpenAI()

def read_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {filepath}: {e}"

def write_file(filepath, content):
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing {filepath}: {e}"

def run_shell_command(command):
    print(f"\n[AGENT EXECUTING]: {command}\n")
    try:
        # 12 minutes max timeout (since the training should take 5 mins max)
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=720)
        output = result.stdout + result.stderr
        
        # Truncate output if it's absurdly long so we don't blow up the context window
        if len(output) > 20000:
            output = output[:10000] + "\n...[OUTPUT TRUNCATED]...\n" + output[-10000:]
            
        return output if output else "Command executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Command timed out after 12 minutes."
    except Exception as e:
        return f"Error executing command: {e}"

# Define the tools available to the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file"}
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write contents to a file. Overwrites the file completely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Full file content"}
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a bash shell command and get the output. Use this for git, uv run, grep, tail, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run"}
                },
                "required": ["command"],
            },
        },
    }
]

def main():
    print("Starting Autonomous Research Agent...")
    
    # 1. Load the system prompt from program.md
    system_prompt = read_file("program.md")
    if system_prompt.startswith("Error"):
        print("Could not load program.md. Make sure you are running this in the autoresearch directory.")
        return

    # 2. Initialize conversation history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hi have a look at program.md and let's kick off a new experiment! let's do the setup first."}
    ]

    # 3. Enter the autonomous loop
    while True:
        try:
            print("\nThinking...")
            response = client.chat.completions.create(
                model="gpt-4o", # You can change this to gpt-4-turbo, etc.
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            messages.append(response_message)

            # If the model wants to call tools, execute them
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n[TOOL CALL] {function_name}({function_args})")
                    
                    if function_name == "read_file":
                        result = read_file(function_args.get("filepath"))
                    elif function_name == "write_file":
                        result = write_file(function_args.get("filepath"), function_args.get("content"))
                    elif function_name == "run_shell_command":
                        result = run_shell_command(function_args.get("command"))
                    else:
                        result = f"Unknown tool: {function_name}"
                        
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": result,
                    })
            else:
                # The model said something directly to us
                print(f"\n[AGENT]: {response_message.content}")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
