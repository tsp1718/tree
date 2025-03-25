import os
import tempfile
import json
import re
import shutil
from fastapi import FastAPI, File, Form, UploadFile, Request, Body,  HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import docx2txt
from pdfminer.high_level import extract_text
# Import LangChain components
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn

load_dotenv()

app = FastAPI()
session_data = {}
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def extract_docx(file_path):
    text = docx2txt.process(file_path)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pdf(file_path):
    text = extract_text(file_path)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_validation_questions_prompt(requirements_text, timeline, user_key_features, free_flow_instruction):
    """
    Generates a prompt to ask for validation questions.

    Inputs:
      - requirements_text: The raw text of the requirements.
      - timeline: The overall project timeline.
      - user_key_features: User provided key features.
      - free_flow_instruction: Additional free flow instruction text.

    The output JSON must exactly follow this structure:
      {
        "qa": [
          { "question": "Question 1 text" },
          { "question": "Question 2 text" },
          ...
        ]
      }
    """
    prompt = f"""

    You are an expert requirements engineer tasked with validating project inputs before creating a solution segment tree. Generate insightful validation questions based on the input provided.

    Inputs:
    - Requirements Text (SRS Document in text format): {requirements_text}
    - Overall Project Timeline: {timeline}
    - User Provided Key Features: {user_key_features}
    - Additional Instructions: {free_flow_instruction}

    ### VALIDATION OBJECTIVES:
    Generate AT MOST 10 strategically targeted questions that:
    1. Clarify ambiguities in the requirements, actors, and key features
    2. Identify potential missing features or capabilities based on the requiremnets_text
    3. Verify the completeness of the defined actors/personas list
    4. Validate that all important features are captured in either the user-defined or system-extracted lists
    5. Check for any inconsistencies or contradictions in the provided inputs
    Your response MUST be in the following JSON format and nothing else:

    ```json
    {{
      "qa": [
        {{
          "question": "Question 1 text"
        }},
        {{
          "question": "Question 2 text"
        }}
        // ... more questions if needed
      ]
    }}
    ```
        ### QUESTION GUIDELINES:
    - Each question should be answerable in under 30 seconds with wither yes/no or just few words
    - Focus on validating existing inputs rather than generating new requirements
    - Prioritize questions that can identify critical issues early
    - Do NOT ask any questions about timelines or project schedules
    - Balance questions across actors, features
    - Format questions to encourage precise, actionable responses
    - Avoid redundant or overlapping questions
    Make sure your output contains ONLY the JSON object and no additional text.
    """
    return prompt.strip()

def ask_questions(
    requirements_text: str,
    user_key_features: str,
    timeline: str,
    free_flow_instruction: str
) -> str:
    """
    Generate supporting questions to build a solution segment tree.

    This function uses:
      - The summary of the requirements document.
      - Key features extracted from the requirements.
      - Additional key features provided by the user.
      - Other inputs like PDF content, project info, actors, workflow stages, timeline, and freeflow input.

    The function returns a string containing a numbered list of supporting questions.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=10000,
        google_api_key=google_api_key
    )

    # Build the complete prompt using the provided inputs.
    full_prompt =  generate_validation_questions_prompt(
        requirements_text=requirements_text,
        user_key_features=user_key_features,
        timeline=timeline,
        free_flow_instruction=free_flow_instruction
    )

    # Create a prompt template with no input variables as the prompt is already complete.
    response = llm.invoke(full_prompt)


    # Return the generated questions as plain text.
    return response.content


def generate_requirements_tree_prompt(requirements_text, timeline, user_key_features, free_flow_instruction, question_answers):
     prompt = f""" You are an expert requirements engineer tasked with generating an extremely detailed and granular requirements tree for a project.

**Inputs:**
- Requirements Text (SRS document in text format): {requirements_text}
- Overall Project Timeline: {timeline}
- User Provided Key Features: {user_key_features}
- Additional Instructions: {free_flow_instruction}
- Answers to Validation Questions: {question_answers}
**Key Guidelines:**  
### 1. Feature and Actor Extraction  
- Extract **each** and **every** possible feature and its potential associated actor(s) from the Requirements Text (SRS) and User-Provided Key Features.   
- If **User-Provided Key Features** differ from those in the **Requirements Text (SRS)**, assign them to the appropriate actor(s) based on the actors defined in the ** Requirements Text (SRS)** and include them as additional entries. Organize them under the appropriate spic group.
- Ensure that **each and every extracted feature and the actor(s) associated with them is always represented** in the final requirements tree.
- Ensure **absolute comprehensiveness** in feature and actor representation.  

### 2. Modularity  
- Organize the requirements tree into multiple **solution segments**, each containing dedicated actor(s), spic groups and the extracted features.  
- Ensure the structure is **extremely modular**, maximizing flexibility, clarity, and ease of maintenance.
### 3. Resolving Contradictions  
**Priority Order:**  
1. **Answers to Validation Questions** (Highest Priority)  
2. **Additional Instructions & User-Provided Key Features**  
3. **Requirements Text (SRS)**  

**Resolution Strategy:**  
- Prefer the **highest-priority source** for each extracted-feature.  
- If contradictions exist, create **separate, explicitly defined** features.  
- **Never merge features** having noticeable differences.

### 4. Timeline Distribution  
- Distribute the overall timeline across **solution segments, actors, spic groups, and features** as follows:  
  - Allocate time **proportionally** based on feature complexity.  
  - Apply **granular weighting** for even the smallest feature components.  
  - Ensure **precise** timeline allocation for each component, with its own dedicated timeline.  
  - Double-check that no component's timeline extends **beyond the overall project timeline**.  
  - Use the strict format: **MM/DD/YYYY - MM/DD/YYYY**.


Your task is to generate an EXHAUSTIVELY DETAILED requirements tree structured with MAXIMUM granularity:
   - Each tree item represents a Solution Segment
   - Each Solution Segment includes:
      - "solution_segment": Extremely specific segment name
      - "timeline": the allocated timeline for this solution segment
      - "actors": Comprehensive list of actors, including:
          - "actor": Precisely defined actor/persona name 
          - "timeline": the allocated timeline for this actor within the segment
          - "spic_groups": Granular groups of related features, including:
              - "spic_group_name": Narrowly defined group name
              - "timeline": the allocated timeline for this spic group within the actor
              - "spics": List of extracted-features, each with:
                  - "spic": Extremely specific feature heading
                  - "timeline": the allocated timeline for this feature within the spic group

Return your response ONLY as a JSON object with MAXIMUM granularity:
```json
{{
  "requirements_tree": [
    {{
      "solution_segment": "Segment Name",
      "timeline": "MM/DD/YYYY - MM/DD/YYYY",
      "actors": [
        {{
          "actor": "Actor Name",
          "timeline": "MM/DD/YYYY - MM/DD/YYYY",
          "spic_groups": [
            {{
              "spic_group_name": "Group Name",
              "timeline": "MM/DD/YYYY - MM/DD/YYYY",
              "spics": [
                {{
                  "spic": "Extracted Feature Heading",
                  "timeline": "MM/DD/YYYY - MM/DD/YYYY"
                }}
                // Extracted feature components
              ]
            }}
            // Additional hyper-specific groups
          ]
        }}
        // Additional precisely defined actors
      ]
    }}
    // Additional solution segments with extreme detail
  ]
}}'''

Ensure extreme detail, modularity, accuracy, and comprehensiveness while creating the requirements tree and strictly follow the JSON structure above."""
     return prompt.strip()


def clean_json_content(content: str) -> str:
        content = content.replace('```json', '').replace('```', '').strip()
            
            # Try to extract JSON content between first and last braces
        json_match = re.search(r'{.*}', content, re.DOTALL | re.MULTILINE)
        if json_match:
                content = json_match.group(0)
            
            # Remove comments and trailing commas
        content = re.sub(r'//.*', '', content)
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*\]', ']', content)
            
        return content.strip()
    

def generate_solution_segment_tree(
    requirements_text: str,
    user_key_features: str,
    timeline: str, 
    free_flow_instruction: str, 
    question_answers: str
) -> dict:
    """Generate solution segment tree with robust JSON parsing"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        max_tokens=10000,
        google_api_key=google_api_key
    )

    # Build and invoke prompt
    full_prompt = generate_requirements_tree_prompt(
        requirements_text=requirements_text,
        user_key_features=user_key_features,
        timeline=timeline,
        free_flow_instruction=free_flow_instruction,
        question_answers=question_answers
    )

    response = llm.invoke(full_prompt)
    
    try:
        # Clean and parse JSON
        cleaned_content = clean_json_content(response.content)
        return json.loads(cleaned_content)

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON Parsing Error: {e}. Raw content: {response.content}")
# Fixed endpoint for handling file uploads
@app.post("/qa")
async def qa(
    srs: UploadFile = File(...),
    timeline: str = Form(...),
    instructions: str = Form(...),
    key_features: str = Form(...)
):
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{srs.filename.split('.')[-1]}") as temp_file:
            # Read content in chunks to handle large files
            content = await srs.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Extract text based on file extension
            extension = srs.filename.split('.')[-1].lower()
            if extension == "pdf":
                text = extract_pdf(temp_file_path)
            elif extension == "docx":
                text = extract_docx(temp_file_path)
            else:
                return JSONResponse(
                    content={"error": f"Unsupported file format: {extension}"},
                    status_code=400
                )

            # Get questions from LLM
            raw_response = ask_questions(
                text,
                key_features,
                timeline,
                instructions
            )

            # Extract JSON from the response
            json_match = re.search(r'```(?:json)?\n(.*?)```', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # If no code blocks, assume the entire response is JSON
                json_str = raw_response.strip()

            # Parse the JSON string into a Python dictionary
            parsed_qa = json.loads(json_str)

            # Store data for the session
            session_data.clear()
            session_data.update({
                "requirements_text": text,
                "timeline": timeline,
                "instructions": instructions,
                "key_features": key_features
            })

            return JSONResponse(
                content={"qa": parsed_qa["qa"]},
                status_code=200
            )

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            content={"error": f"Internal Server Error: {str(e)}", "details": error_details},
            status_code=500
        )

@app.post("/tree")
async def tree(qa_data: dict):
    try:
        if not session_data:
            return JSONResponse(
                content={"error": "No active session. Please call /qa/ first."}, 
                status_code=400
            )
        
        # Validate that the incoming data has the expected structure
        if "qa" not in qa_data or not isinstance(qa_data["qa"], list):
            return JSONResponse(
                content={"error": "Invalid request format. Expected {'qa': []} structure."}, 
                status_code=400
            )
        


        formatted_qa = json.dumps(qa_data)
        
        tree = generate_solution_segment_tree(
            session_data["requirements_text"],
            session_data["key_features"],
            session_data["timeline"],
            session_data["instructions"],
            formatted_qa  # Passing the formatted QA data
        )

        return JSONResponse(content=tree)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            content={"error": f"Internal Server Error: {str(e)}", "details": error_details},
            status_code=500
        )
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
