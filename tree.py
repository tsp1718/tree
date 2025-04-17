# Import necessary libraries
import os
import tempfile
import json
import re
import shutil
from typing import Optional, Dict, Any # Added Optional for type hinting

# Import FastAPI components
from fastapi import FastAPI, File, Form, UploadFile, Request, Body, HTTPException # UploadFile is needed for type hints
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import utility libraries
from dotenv import load_dotenv
import docx2txt
from pdfminer.high_level import extract_text
from pathlib import Path
import uvicorn
import traceback # Import traceback for detailed error logging

# Import LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables (e.g., API keys) from a .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Requirements Analysis API")

# Global dictionary to store session data between /qa and /tree calls
session_data: Dict[str, Any] = {}

# Add CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Helper Functions ---

# Corrected: Input type hint is UploadFile, return type hint is str
# Corrected: Used f.filename in error message
def extract_txt(f: UploadFile) -> str:
    try:
        # Read directly from the file-like object provided by UploadFile
        # Use await f.read() if using async libraries, but standard read() works here
        # Need to decode bytes to string
        content_bytes = f.file.read()
        text = content_bytes.decode("utf-8")
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        # Log the error or handle it more gracefully
        print(f"Error extracting TXT from {f.filename}: {e}") # Corrected variable
        raise HTTPException(status_code=500, detail=f"Failed to process TXT file: {e}")
    finally:
        # It's good practice to ensure the file stream is closed,
        # though FastAPI often handles this.
        # However, libraries like pdfminer might leave it open in case of error.
        if f.file and not f.file.closed:
            f.file.close()


# Corrected: Input type hint is UploadFile, return type hint is str, removed extra colon
# Corrected: Used f.filename in error message
def extract_docx(f: UploadFile) -> str:
    try:
        # docx2txt can process file-like objects directly
        text = docx2txt.process(f.file)
        # Replace multiple whitespace characters with a single space and strip leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        # Log the error or handle it more gracefully
        print(f"Error extracting DOCX from {f.filename}: {e}") # Corrected variable
        raise HTTPException(status_code=500, detail=f"Failed to process DOCX file: {e}")
    finally:
        if f.file and not f.file.closed:
            f.file.close()


# Corrected: Input type hint is UploadFile, return type hint is str, removed extra colons
# Corrected: Used f.filename in error message
def extract_pdf(f: UploadFile) -> str:
    try:
        # pdfminer's extract_text works with file-like objects
        text = extract_text(f.file)
        # Replace multiple whitespace characters with a single space and strip leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        # Log the error or handle it more gracefully
        print(f"Error extracting PDF from {f.filename}: {e}") # Corrected variable
        raise HTTPException(status_code=500, detail=f"Failed to process PDF file: {e}")
    finally:
        if f.file and not f.file.closed:
            f.file.close()


def generate_validation_questions_prompt(
    requirements_text: str,
    timeline: str,
    user_key_features: Optional[str],
    free_flow_instruction: Optional[str]
) -> str:
    # Provide default values for optional fields if they are None or empty for the prompt
    key_features_prompt = user_key_features if user_key_features else "Not provided."
    instructions_prompt = free_flow_instruction if free_flow_instruction else "Not provided."

    prompt = f"""
    You are an expert requirements engineer tasked with validating project inputs before creating a solution segment tree. Generate insightful validation questions based on the input provided.

    Inputs:
    - Requirements Text (SRS Document in text format): {requirements_text}
    - Overall Project Timeline: {timeline}
    - User Provided Key Features: {key_features_prompt}
    - Additional Instructions: {instructions_prompt}

    ### VALIDATION OBJECTIVES:
    Generate AT MOST 10 strategically targeted questions that:
    1. Clarify ambiguities in the requirements, actors, and key features.
    2. Identify potential missing features or capabilities based on the requirements_text.
    3. Verify the completeness of the defined actors/personas list.
    4. Validate that all important features are captured in either the user-defined or system-extracted lists.
    5. Check for any inconsistencies or contradictions in the provided inputs.

    Your response MUST be in the following JSON format and nothing else:

    ```json
    {{
      "qa": [
        {{ "question": "Question 1 text" }},
        {{ "question": "Question 2 text" }}
        // ... more questions if needed (up to 10 total)
      ]
    }}
    ```

    ### QUESTION GUIDELINES:
    - Each question should be answerable in under 30 seconds with either yes/no or just a few words.
    - Focus on validating existing inputs rather than generating new requirements.
    - Prioritize questions that can identify critical issues early.
    - Do NOT ask any questions about timelines or project schedules.
    - Balance questions across actors and features.
    - Format questions to encourage precise, actionable responses.
    - Avoid redundant or overlapping questions.
    - Ensure your output contains ONLY the JSON object and no additional text or markdown formatting.
    """
    return prompt.strip()


def ask_questions(
    requirements_text: str,
    timeline: str,
    user_key_features: Optional[str],
    free_flow_instruction: Optional[str]
) -> str:

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")

    try:
        # Initialize the Google Generative AI model
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=10000, # Adjust as needed
            google_api_key=google_api_key,
            # Consider adding response_mime_type="application/json" if supported
            # and the model consistently returns valid JSON. This can help parsing.
        )

        # Build the complete prompt using the provided inputs.
        full_prompt = generate_validation_questions_prompt(
            requirements_text=requirements_text,
            timeline=timeline,
            user_key_features=user_key_features,
            free_flow_instruction=free_flow_instruction
        )

        # Invoke the LLM with the prompt.
        response = llm.invoke(full_prompt)

        # Return the generated content (expected to be JSON).
        return response.content

    except Exception as e:
        # Log the error for debugging
        print(f"Error during LLM call in ask_questions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions from LLM: {e}")


def generate_requirements_tree_prompt(
    requirements_text: str,
    timeline: str,
    user_key_features: Optional[str],
    free_flow_instruction: Optional[str],
    Youtubes: str
) -> str:

    # Provide default values for optional fields if they are None or empty for the prompt
    key_features_prompt = user_key_features if user_key_features else "Not provided."
    instructions_prompt = free_flow_instruction if free_flow_instruction else "Not provided."

    prompt = f"""
    You are an expert requirements engineer tasked with generating an extremely detailed and granular requirements tree for a project based on the provided inputs and validated answers.

    **Inputs:**
    - Requirements Text (SRS document in text format): {requirements_text}
    - Overall Project Timeline: {timeline}
    - User Provided Key Features: {key_features_prompt}
    - Additional Instructions: {instructions_prompt}
    - Answers to Validation Questions (JSON format): {Youtubes}

    **Key Guidelines:**

    ### 1. Feature and Actor Extraction
    - Extract **each** and **every** possible feature and its potential associated actor(s) from the **Requirements Text (SRS)** and **User-Provided Key Features**.
    - Use the **Answers to Validation Questions** to clarify ambiguities or confirm details about features and actors.
    - If **User-Provided Key Features** differ from those in the **Requirements Text (SRS)**, assign them to the appropriate actor(s) based on the actors defined in the **Requirements Text (SRS)** and include them as additional entries. Organize them under the appropriate spic group.
    - Ensure that **each and every extracted feature and the actor(s) associated with them is always represented** in the final requirements tree.
    - Ensure **absolute comprehensiveness** in feature and actor representation.

    ### 2. Modularity
    - Organize the requirements tree into multiple **solution segments**, each containing dedicated actor(s), spic groups, and the extracted features.
    - Ensure the structure is **extremely modular**, maximizing flexibility, clarity, and ease of maintenance.

    ### 3. Resolving Contradictions
    **Priority Order:**
    1. **Answers to Validation Questions** (Highest Priority)
    2. **Additional Instructions & User-Provided Key Features**
    3. **Requirements Text (SRS)**

    **Resolution Strategy:**
    - Prefer the **highest-priority source** for each extracted feature.
    - If contradictions persist even after considering the answers, create **separate, explicitly defined** features reflecting the different sources.
    - **Never merge features** having noticeable differences unless explicitly clarified by the answers.

    ### 4. Timeline Distribution
    - Distribute the overall timeline ({timeline}) across **solution segments, actors, spic groups, and features** as follows:
      - Allocate time **proportionally** based on estimated feature complexity and dependencies identified or implied in the inputs/answers.
      - Apply **granular weighting** for even the smallest feature components.
      - Ensure **precise** timeline allocation for each component, with its own dedicated timeline range.
      - Double-check that no component's timeline extends **beyond the overall project timeline**.
      - Use the strict format: **MM/DD/YYYY - MM/DD/YYYY**.

    Your task is to generate an EXHAUSTIVELY DETAILED requirements tree structured with MAXIMUM granularity. When creating the "spic" entries, do not include any reference codes in parentheses (such as U-04, AD-06, etc.) in the spic value text. Return your response ONLY as a JSON object following this exact structure:

    ```json
    {{
      "requirements_tree": [
        {{
          "solution_segment": "Segment Name (e.g., User Authentication)",
          "timeline": "MM/DD/YYYY - MM/DD/YYYY", // Segment-level timeline
          "actors": [
            {{
              "actor": "Actor Name (e.g., Registered User)",
              "timeline": "MM/DD/YYYY - MM/DD/YYYY", // Actor-level timeline within segment
              "spic_groups": [
                {{
                  "spic_group_name": "Group Name (e.g., Login Functionality)",
                  "timeline": "MM/DD/YYYY - MM/DD/YYYY", // Group-level timeline within actor
                  "spics": [
                    {{
                      "spic": "Extracted Specific Feature Heading (e.g., Implement password hashing)",
                      "timeline": "MM/DD/YYYY - MM/DD/YYYY" // Feature-level timeline
                    }}
                    // ... other highly specific features within this group
                  ]
                }}
                // ... other spic groups for this actor
              ]
            }}
            // ... other actors within this segment
          ]
        }}
        // ... other solution segments
      ]
    }}
    ```

    Ensure extreme detail, modularity, accuracy, and comprehensiveness while creating the requirements tree. Strictly adhere to the JSON structure above and provide ONLY the JSON object in your response.
    """
    return prompt.strip()


def clean_json_content(content: str) -> str:
        # Remove potential markdown formatting around the JSON
        content = content.replace('```json', '').replace('```', '').strip()

        # Try to extract JSON content between first and last curly braces
        # This helps if the LLM adds introductory/closing text around the JSON block
        json_match = re.search(r'\{.*\}', content, re.DOTALL | re.MULTILINE)
        if json_match:
                content = json_match.group(0)
        # If no match, assume the whole content *might* be JSON and proceed

        # Remove single-line comments (LLM sometimes adds them)
        content = re.sub(r'//.*', '', content)
        # Remove multi-line comments (less common, but possible)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Remove trailing commas before closing braces and brackets (common JSON error)
        content = re.sub(r',\s*\}', '}', content)
        content = re.sub(r',\s*\]', ']', content)

        return content.strip()


def generate_solution_segment_tree(
    requirements_text: str,
    timeline: str,
    user_key_features: Optional[str],
    free_flow_instruction: Optional[str],
    Youtubes: str # JSON string
) -> dict:

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set.")

    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            max_tokens=25000, # Increased max_tokens for potentially large tree
            google_api_key=google_api_key,
            # Consider adding response_mime_type="application/json" here too
        )

        # Build the prompt for generating the tree
        full_prompt = generate_requirements_tree_prompt(
            requirements_text=requirements_text,
            timeline=timeline,
            user_key_features=user_key_features,
            free_flow_instruction=free_flow_instruction,
            Youtubes=Youtubes
        )

        # Invoke LLM
        response = llm.invoke(full_prompt)
        raw_content = response.content

        # Clean the response content to isolate JSON
        cleaned_content = clean_json_content(raw_content)

        # Parse the cleaned JSON string
        parsed_tree = json.loads(cleaned_content)
        return parsed_tree

    except json.JSONDecodeError as e:
        # Log the error and the problematic content for debugging
        print(f"JSON Parsing Error: {e}")
        print(f"--- Raw LLM content ---:\n{raw_content[:1000]}...") # Log start of raw content
        print(f"--- Cleaned content before parsing ---:\n{cleaned_content[:1000]}...") # Log start of cleaned content
        # Raise HTTPException to inform the client
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse the requirements tree JSON from LLM response. Error: {e}. Please check the LLM output format."
        )
    except Exception as e:
        # Catch other potential errors during LLM call or processing
        print(f"Error during LLM call/processing in generate_solution_segment_tree: {e}")
        traceback.print_exc() # Print full traceback for server logs
        raise HTTPException(status_code=500, detail=f"Failed to generate requirements tree: {e}")


# --- API Endpoints ---

@app.post("/qa")
async def qa(
    # --- Mandatory Parameters ---
    srs: UploadFile = File(..., description="The SRS document (PDF, DOCX, or TXT)"),
    timeline: str = Form(..., description="Overall project timeline"),

    # --- Optional Parameters ---
    instructions: Optional[str] = Form(None, description="Optional additional instructions for the analysis"),
    key_features: Optional[str] = Form(None, description="Optional comma-separated list of key features provided by the user")
):
    text = "" # Initialize text
    # Ensure filename is not None before proceeding
    if srs.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")

    # Determine file type and extract text
    try:
        if srs.filename.lower().endswith('.pdf'):
            text = extract_pdf(srs)
        elif srs.filename.lower().endswith('.docx'):
            text = extract_docx(srs)
        elif srs.filename.lower().endswith('.txt'):
            text = extract_txt(srs)
        else:
            # Correctly get extension for error message
            extension = srs.filename.split('.')[-1]
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: '.{extension}'. Please upload a PDF, DOCX, or TXT file."
            )

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from the uploaded document. It might be empty or corrupted.")

        # --- Call LLM to get questions ---
        raw_response = ask_questions(
            requirements_text=text,
            timeline=timeline,
            user_key_features=key_features, # Pass optional value
            free_flow_instruction=instructions # Pass optional value
        )

        # --- Process LLM Response ---
        cleaned_json_str = "" # Initialize
        try:
            # Clean the response to get potential JSON
            cleaned_json_str = clean_json_content(raw_response)
            # Parse the JSON string into a Python dictionary
            parsed_qa = json.loads(cleaned_json_str)

            # Validate the structure of the parsed JSON
            if "qa" not in parsed_qa or not isinstance(parsed_qa.get("qa"), list):
                raise ValueError("LLM response missing 'qa' list structure or 'qa' is not a list.")

        except (json.JSONDecodeError, ValueError) as e:
            # Log the error and the problematic content
            print(f"JSON Parsing/Validation Error in /qa endpoint: {e}")
            print(f"--- Raw LLM response ---:\n{raw_response[:1000]}...")
            print(f"--- Cleaned content before parsing ---:\n{cleaned_json_str[:1000]}...")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse validation questions JSON from LLM response. Error: {e}"
            )

        # --- Store data for the next step (/tree) ---
        # Clear previous session data and store new data
        # Consider using a more robust session mechanism for production
        # (e.g., client-side tokens, server-side session store like Redis)
        # if this needs to scale or handle concurrent users properly.
        session_data.clear()
        session_data.update({
            "requirements_text": text,
            "timeline": timeline,
            "instructions": instructions,      # Store optional value (can be None)
            "key_features": key_features       # Store optional value (can be None)
        })

        # Return the questions to the client
        return JSONResponse(
            content={"qa": parsed_qa["qa"]}, # Return only the list of questions
            status_code=200
        )

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during file processing or LLM calls
        error_details = traceback.format_exc()
        print(f"Internal Server Error in /qa: {e}\n{error_details}") # Log detailed error
        # Ensure file stream is closed in case of unexpected error during processing
        if srs.file and not srs.file.closed:
            await srs.close() # Use await for async close
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}" # Send a generic message to client
        )
    finally:
         # Ensure the uploaded file stream is closed after processing in /qa
         # This is important as FastAPI might not close it automatically if
         # an exception occurs *before* the response is sent, or if reads
         # didn't consume the whole stream.
        if srs.file and not srs.file.closed:
             await srs.close() # Use await for async close


@app.post("/tree")
async def tree(qa_data: dict = Body(..., description="JSON object containing the list of questions and user answers, e.g., {'qa': [{'question': '...', 'answer': '...'}, ...]}")):
    try:
        # --- Validate Session and Input ---
        if not session_data:
            raise HTTPException(
                status_code=400, # Bad Request, as the prerequisite /qa call wasn't made or failed
                detail="No active session found. Please call the /qa endpoint first to upload the SRS document and get questions."
            )

        # Validate the structure of the incoming qa_data
        if "qa" not in qa_data or not isinstance(qa_data.get("qa"), list):
            raise HTTPException(
                status_code=422, # Unprocessable Entity, input format is wrong
                detail="Invalid request format. Expected a JSON body with a 'qa' key containing a list of question-answer objects, like {'qa': [{'question': '...', 'answer': '...'}]}"
            )
        # Optional: Add further validation for 'question' and 'answer' keys within the list items
        # for item in qa_data.get("qa", []):
        #     if not isinstance(item, dict) or "question" not in item or "answer" not in item:
        #          raise HTTPException(status_code=422, detail="Invalid item format in 'qa' list. Each item must be a dictionary with 'question' and 'answer' keys.")


        # Format the QA data into a JSON string for the LLM prompt
        # Ensure proper JSON formatting for the prompt
        formatted_qa_string = json.dumps(qa_data, indent=2) # Use indent for readability in logs/prompts if needed

        # --- Retrieve data from session ---
        # Use .get() to safely retrieve data
        requirements_text = session_data.get("requirements_text")
        timeline = session_data.get("timeline")
        instructions = session_data.get("instructions") # Might be None
        key_features = session_data.get("key_features") # Might be None

        # Check if mandatory data is present (should be if session_data is not empty)
        if not requirements_text or not timeline:
             # Log this server-side, indicates a state management issue
             print("Error: Session data missing requirements_text or timeline in /tree endpoint.")
             raise HTTPException(status_code=500, detail="Session data is corrupted or incomplete. Please try the /qa step again.")


        # --- Generate Tree ---
        generated_tree = generate_solution_segment_tree(
            requirements_text=requirements_text,
            timeline=timeline,
            user_key_features=key_features, # Pass optional value
            free_flow_instruction=instructions, # Pass optional value
            Youtubes=formatted_qa_string # Pass the JSON string of Q&A
        )


        # Return the generated tree
        # FastAPI automatically converts the dict to a JSON response
        return generated_tree # Return the dictionary directly

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during tree generation or processing
        error_details = traceback.format_exc()
        print(f"Internal Server Error in /tree: {e}\n{error_details}") # Log detailed error
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while generating the tree: {str(e)}"
        )

# --- Run the application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
