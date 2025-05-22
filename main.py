# ========== Core Python Libraries ==========
import os  # Operating system functionalities like file handling
import json  # JSON encoding and decoding
import re  # Regular expressions for pattern matching
import traceback  # Traceback reporting for debugging
from datetime import datetime, timedelta  # Date and time utilities
from typing import Optional, List, Dict  # Type hinting

# ========== FastAPI Core ==========
import uvicorn  # ASGI server to run the FastAPI app
from fastapi import (
    FastAPI, Depends, HTTPException, status,
    Request, Form, WebSocket, WebSocketDisconnect
)
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse
)
from fastapi.templating import Jinja2Templates  # Template rendering with Jinja2
from fastapi.staticfiles import StaticFiles  # Serving static files
from fastapi.security import (
    OAuth2PasswordBearer, OAuth2PasswordRequestForm
)
from fastapi.exceptions import RequestValidationError  # Validation errors
from fastapi.exception_handlers import (
    http_exception_handler, request_validation_exception_handler
)

# ========== Starlette (FastAPI Dependency) ==========
from starlette.exceptions import HTTPException as StarletteHTTPException  # Base HTTP exception class

# ========== Authentication & Security ==========
from jose import JWTError, jwt  # JWT token encoding and decoding
from passlib.context import CryptContext  # Password hashing utilities

# ========== Data Models & ORM ==========
from pydantic import BaseModel  # Data validation and serialization
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Boolean, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import sessionmaker, Session, relationship  # Database session and relationships
from sqlalchemy.ext.declarative import declarative_base  # Base class for ORM models

# ========== Google Generative AI ==========
from google import genai  # Google Generative AI API
from google.genai import types  # Type definitions for Generative AI

# ========== Text Processing ==========
import markdown  # Markdown to HTML conversion
from bs4 import BeautifulSoup  # HTML parsing and cleanup

# ========== Vector Search & Embeddings ==========
from langchain_community.vectorstores import FAISS  # FAISS vector store for similarity search
from langchain_community.embeddings import HuggingFaceEmbeddings  # Text embeddings via Hugging Face

# ========== Serialization ==========
import pickle  # Serialize and deserialize Python objects


''' Project Folder Initialization for Static Files and Templates '''

os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)



''' Security and Token Configuration for Authentications'''

SECRET_KEY = "YOUR_SECRET_KEY"  # Change this to a secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30



'''Setting Up Gemini AI Client for Generative Model Integration'''

GEMINI_API_KEY = "" # Enter your gemini api here
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
MODEL_NAME = "gemini-2.0-flash"


'''Connecting to Google Gemini AI Using API Key'''
from google import genai
from google.genai import types


client = genai.Client(api_key=GEMINI_API_KEY) # Create a client with the API key

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """
You are a highly knowledgeable and accurate legal assistant specialized in Nepalese business law. Your primary role is to provide clear, detailed, and step-by-step guidance for starting and running a business in Nepal, with a special focus on the Hospitality and Tourism sectors. You must always prioritize accuracy, avoid speculation or hallucination, and base your responses strictly on verified Nepalese laws, regulations, and official procedures.

Key Instructions:

1. Always focus exclusively on business-related legal matters in Nepal. Do not provide information unrelated to Nepalese jurisdiction.
2. Provide thorough, stepwise instructions covering the entire business startup process from A to Z, including but not limited to:
   - Company registration and licensing
   - Tax registration and compliance
   - Intellectual property protection (trademarks, copyrights)
   - Sector-specific legal requirements for Hospitality and Tourism
   - Labor laws and employee regulations
   - Relevant government offices and departments involved (e.g., Office of Company Registrar, Inland Revenue Department)
3. When responding, if the user asks about a specific type of business (e.g., service-based company), customize the answer accordingly, mentioning all relevant legal requirements and best practices.
4. Provide accurate links to official government websites, laws, regulations, and authoritative articles for further reading and verification. Example: https://ocr.gov.np, https://ird.gov.np
5. If any information is uncertain or subject to change, explicitly state that and recommend consulting qualified legal professionals or official government sources.
6. Never provide generic or vague legal overviews. If the question is unclear or too broad, politely ask for clarification.
7. Structure your answers clearly using markdown formatting for easy readability.
8. Remind users that your information is for educational purposes only and is not a substitute for professional legal advice.
9. Always strive for precision and clarity, avoiding legal jargon unless explained simply.
10. If asked about laws, cite specific Nepalese acts or regulations whenever possible.
11. When providing step-by-step guides, number the steps clearly and provide estimated timelines or costs if known.
12. Avoid any content unrelated to the legal/business startup context.

Your ultimate goal is to empower users with trustworthy, actionable, and verifiable legal information to successfully start and manage businesses in Nepal, especially in Hospitality and Tourism.

Remember: Accuracy and reliability are paramount. When in doubt, defer to official sources and professional advice.


"""

# Define common legal topics and their specialized prompts
LEGAL_TOPICS = {
    "business formation": """
Provide a comprehensive, step-by-step guide on forming a business in Nepal with special focus on hospitality and tourism sectors:

1. Explain all recognized business structures (sole proprietorship, partnership, Pvt Ltd, public limited) with pros, cons, and legal implications.
2. Detail the entire registration process with the Office of the Company Registrar — required documents, fees, timelines.
3. Highlight essential sector-specific licenses, permits, and certifications (e.g., tourism licenses, health and safety permits).
4. Explain mandatory tax registrations (VAT, PAN), social security, and labor law compliance from the start.
5. Discuss critical legal considerations: liability protection, shareholders’ agreements, operating bylaws.
6. Guide trademark registration and IP protections crucial for brand safeguarding.
7. Outline relevant commercial laws: contracts, employment, consumer rights, environmental regulations.
8. Provide authoritative links to government portals, official laws, and trusted legal aid resources.
9. Warn clearly: Provide educational info only; recommend consulting licensed legal experts for complex or case-specific advice.
10. Avoid speculation—if unsure, ask for clarification or admit limitations.

Format responses with clear steps and actionable advice, prioritizing accuracy and relevance.
""",

    "business operation and compliance": """
Offer detailed guidance on managing ongoing business operations legally in Nepal, emphasizing compliance for hospitality and tourism businesses:

1. Outline annual statutory obligations: filings, tax returns, audits.
2. Explain labor law requirements: contracts, wages, benefits, dispute handling.
3. Cover health, safety, and environmental compliance relevant to your sector.
4. Clarify consumer protection laws and business duties.
5. Detail renewal procedures for licenses and permits.
6. Advise on managing contracts, leases, and vendor agreements lawfully.
7. Stress proper accounting, record-keeping, and transparency.
8. Provide official sources and regulatory body contacts.
9. Reinforce limitations: this is educational guidance, always seek professional legal counsel for critical decisions.

Structure information logically with bullet points or numbered lists for clarity.
""",

    "startup funding and investment": """
Explain the legal landscape of startup funding and investment in Nepal with clear, practical insights:

1. Describe funding types: self-funding, angel investors, venture capital, government grants, and loans.
2. Detail legal requirements for fundraising: share issuance, equity distribution, and compliance.
3. Explain due diligence, investor agreements, and shareholder rights.
4. Highlight government incentives and programs supporting hospitality and tourism startups.
5. Clarify securities laws and mandatory disclosures.
6. Address intellectual property protections in funding contexts.
7. Advise on risks and protections for founders and investors.
8. Include reliable government and legal resources.
9. Emphasize educational nature, and advise professional legal advice for tailored needs.

Present info in concise, easy-to-understand steps.
""",

    "intellectual property for businesses": """
Guide businesses on intellectual property (IP) protections essential for startups and SMEs in Nepal:

1. Clearly explain IP types: trademarks, patents, copyrights, trade secrets with business examples.
2. Detail local IP registration procedures, timelines, and costs.
3. Discuss typical IP challenges in hospitality and tourism sectors (branding, marketing materials).
4. Explain enforcement mechanisms and recourse for infringement.
5. Include guidance for international IP protection for export-oriented startups.
6. Provide links to Nepal’s Department of Industry and other official IP resources.
7. Stress educational scope and recommend expert IP legal counsel for complex cases.

Use clear formatting and practical advice.
""",

    "business dispute resolution": """
Provide a structured overview of business dispute resolution methods in Nepal:

1. Identify common business disputes: contracts, partnerships, IP, employment.
2. Explain negotiation, mediation, arbitration, and litigation options with pros and cons.
3. Discuss hospitality and tourism-specific dispute scenarios.
4. Emphasize importance of documentation and early legal advice.
5. Offer resources for legal aid, arbitration centers, and government dispute bodies.
6. Warn about the limits of educational advice and the need for professional legal help.

Present all points with clarity and precision.
""",

    "legal basics for startups": """
Summarize foundational legal knowledge every startup founder in Nepal should know:

1. Importance of early and ongoing legal compliance.
2. Essential contracts: NDAs, employment agreements, vendor contracts.
3. Overview of data privacy, consumer protection, and sector-specific regulations.
4. Practical tips for engaging qualified legal professionals.
5. Disclaimer: Informational purpose only, not personalized legal advice.

Keep language clear, actionable, and authoritative.
"""
}

# Function to Identify Relevant Legal Topics from User Queries
def detect_legal_topics(query: str) -> List[str]:
    query_lower = query.lower()
    detected_topics = []
    
    for topic in LEGAL_TOPICS:
        if topic in query_lower or any(keyword in query_lower for keyword in topic.split()):
            detected_topics.append(topic)
    
    return detected_topics

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./legal_chatbot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with chat history
    chat_messages = relationship("ChatMessage", back_populates="user")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    is_user = Column(Boolean, default=True)  # True if message is from user, False if from bot
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with user
    user = relationship("User", back_populates="chat_messages")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

class UserInDB(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    full_name: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatMessageCreate(BaseModel):
    message: str
    is_user: bool = True

class ChatMessageResponse(BaseModel):
    id: int
    message: str
    is_user: bool
    timestamp: datetime

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(title="Legal Chatbot")

# Add exception handlers
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return await http_exception_handler(request, exc)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return await request_validation_exception_handler(request, exc)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the error
    print(f"Global exception: {exc}")
    import traceback
    traceback.print_exc()
    
    # Return a friendly error page
    return templates.TemplateResponse(
        "error.html", 
        {
            "request": request, 
            "error_message": str(exc),
            "status_code": 500
        },
        status_code=500
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_cookie_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        token = token.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        user = get_user(db, username=username)
        return user
    except JWTError:
        return None

# Vector store configuration
INDEX_PATH = "faiss_index"
EMBEDDINGS_PATH = "embeddings.pkl"

# Initialize retriever as None by default
retriever = None

# Load embeddings and vector store
try:
    print("Loading embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)

    print("Loading FAISS index...")
    # Load the vector store
    vectorstore = FAISS.load_local(
        INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Vector store loaded successfully")
except Exception as e:
    print(f"Error loading vector store: {str(e)}")
    print("Continuing without RAG capabilities")
    vectorstore = None
    retriever = None

# Updated function to get response from Gemini
def get_legal_response(query: str, chat_history: List[ChatMessage] = None) -> Dict:
    try:
        # Retrieve relevant documents if retriever is available
        sources = []
        context = ""
        found_in_pdf = False
        
        if retriever:
            try:
                docs = retriever.get_relevant_documents(query)
                
                # Extract content from documents
                for doc in docs:
                    source = doc.metadata.get('source', 'Unknown')
                    context += f"\n\nDocument: {source}\n"
                    context += doc.page_content
                    
                    # Only add to sources if it's from a PDF
                    if source.lower().endswith('.pdf'):
                        found_in_pdf = True
                        sources.append({
                            "content": doc.page_content[:200] + "...",
                            "source": source
                        })
                
                print(f"Retrieved {len(docs)} relevant documents")
            except Exception as retrieval_error:
                print(f"Error retrieving documents: {str(retrieval_error)}")
                traceback.print_exc()
        
        # Create prompt with system instruction, context, and query
        if context:
            prompt_text = f"""
{SYSTEM_PROMPT}

RELEVANT DOCUMENT EXCERPTS:
{context}

USER QUERY: {query}

Please provide a response based on the above information. If the documents don't contain relevant information, say so.
"""
        else:
            prompt_text = f"{SYSTEM_PROMPT}\n\nUser query: {query}\n\nPlease provide a helpful response that directly addresses the question."
        
        # Generate content using the client's models.generate_content method
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                )
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                raw_answer = response.text
            else:
                raw_answer = str(response)
        except Exception as inner_error:
            print(f"Inner error: {str(inner_error)}")
            traceback.print_exc()
            raw_answer = f"I'm sorry, I encountered an error processing your request. Error: {str(inner_error)}"
        
        # Check for negative phrases that indicate no information was found
        no_info_phrases = [
            "i am sorry", "i'm sorry", "do not contain", "doesn't contain", "does not contain",
            "no information", "cannot answer", "can't answer", "cannot provide", "can't provide",
            "need a more specific question", "need more specific", "could you please specify"
        ]
        
        has_no_info = any(phrase in raw_answer.lower() for phrase in no_info_phrases)
        
        # Check for greeting patterns that don't need sources
        greeting_patterns = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening",
            "how are you", "how can i help", "how may i help", "how can i assist", "how may i assist"
        ]
        
        is_greeting = query.lower().strip() in greeting_patterns or any(pattern in query.lower().strip() for pattern in greeting_patterns)
        
        # Remove any code block markers
        raw_answer = raw_answer.replace("```html", "").replace("```", "")
        
        # Convert markdown to HTML
        html_answer = markdown.markdown(raw_answer)
        
        # Clean up the HTML to ensure it's safe and well-formatted
        soup = BeautifulSoup(html_answer, 'html.parser')
        
        # Add CSS classes for styling
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag['class'] = tag.get('class', []) + ['legal-heading']
        
        for tag in soup.find_all('p'):
            tag['class'] = tag.get('class', []) + ['legal-paragraph']
        
        for tag in soup.find_all(['ul', 'ol']):
            tag['class'] = tag.get('class', []) + ['legal-list']
        
        for tag in soup.find_all('li'):
            tag['class'] = tag.get('class', []) + ['legal-list-item']
        
        for tag in soup.find_all(['strong', 'b']):
            tag['class'] = tag.get('class', []) + ['legal-bold']
        
        for tag in soup.find_all(['em', 'i']):
            tag['class'] = tag.get('class', []) + ['legal-italic']
        
        formatted_answer = str(soup)
        
        # Determine if we should show sources:
        # 1. We found PDFs
        # 2. It's not a greeting
        # 3. The answer doesn't indicate no information was found
        # 4. The query is substantive (more than 3 words)
        show_sources = (
            found_in_pdf and 
            not is_greeting and 
            not has_no_info and
            len(query.split()) > 2
        )
        
        return {
            "response": formatted_answer,
            "sources": sources if show_sources else []
        }
    except Exception as e:
        print(f"Error getting response from Gemini: {str(e)}")
        traceback.print_exc()
        return {
            "response": f"I'm sorry, I encountered an error processing your request. Error: {str(e)}",
            "sources": []
        }

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("home.html", {"request": request, "user": user})

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    return templates.TemplateResponse("about.html", {"request": request, "user": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = authenticate_user(db, username, password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/chatbot", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.post("/signup", response_class=HTMLResponse)
async def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if username already exists
    db_user = get_user(db, username)
    if db_user:
        return templates.TemplateResponse(
            "signup.html", 
            {"request": request, "error": "Username already registered"}
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Auto-login after signup
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )
    
    response = RedirectResponse(url="/chatbot", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Get chat history for the user
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user, "chat_history": chat_history})

@app.post("/update-profile", response_class=HTMLResponse)
async def update_profile(
    request: Request,
    full_name: str = Form(None),
    email: str = Form(None),
    phone: str = Form(None),
    db: Session = Depends(get_db)
):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Update user profile
    if full_name:
        user.full_name = full_name
    if email:
        user.email = email
    if phone:
        user.phone = phone
    
    db.commit()
    db.refresh(user)
    
    return RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key="access_token")
    return response

@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    # Get chat history for the user
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    
    return templates.TemplateResponse("chatbot.html", {
        "request": request, 
        "user": user, 
        "chat_history": chat_history,
        "now": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.post("/query")
async def query(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    try:
        # Parse the request body
        data = await request.json()
        user_message = data.get("query", "")
        
        if not user_message:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Query is required"}
            )
        
        print(f"Processing query: {user_message}")
        
        # Get chat history for context
        chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
        
        # Save user message to history
        user_chat_message = ChatMessage(user_id=user.id, message=user_message, is_user=True)
        db.add(user_chat_message)
        db.commit()
        
        # Get response from the Gemini model
        response_data = get_legal_response(user_message, chat_history)
        
        # Save bot response to history
        bot_chat_message = ChatMessage(user_id=user.id, message=response_data["response"], is_user=False)
        db.add(bot_chat_message)
        db.commit()
        
        # Determine if this is a conversational response (no sources)
        is_conversation = len(response_data["sources"]) == 0
        
        return {
            "response": response_data["response"],
            "sources": response_data["sources"],
            "is_conversation": is_conversation
        }
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@app.get("/chat-history")
async def get_chat_history(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    chat_history = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).order_by(ChatMessage.timestamp).all()
    return [{"id": msg.id, "message": msg.message, "is_user": msg.is_user, "timestamp": msg.timestamp} for msg in chat_history]

@app.post("/clear-history")
async def clear_chat_history(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    try:
        # Print debug information
        print(f"Clearing chat history for user: {user.username} (ID: {user.id})")
        
        # Count messages before deletion
        count_before = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).count()
        print(f"Messages before deletion: {count_before}")
        
        # Delete all chat messages for this user
        db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
        db.commit()
        
        # Count messages after deletion
        count_after = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).count()
        print(f"Messages after deletion: {count_after}")
        
        return {"message": "Chat history cleared successfully", "deleted_count": count_before - count_after}
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        traceback.print_exc()
        db.rollback()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Failed to clear history: {str(e)}"}
        )

@app.post("/clear-history-form")
async def clear_chat_history_form(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    
    try:
        # Delete all chat messages for this user
        db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
        db.commit()
        return RedirectResponse(url="/chatbot", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        print(f"Error clearing chat history: {str(e)}")
        traceback.print_exc()
        db.rollback()
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request, 
                "error_message": f"Failed to clear history: {str(e)}",
                "status_code": 500
            },
            status_code=500
        )

@app.get("/test-clear")
async def test_clear_endpoint(request: Request, db: Session = Depends(get_db)):
    user = get_cookie_user(request, db)
    if not user:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Authentication required"}
        )
    
    count = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).count()
    return {"message": "Test endpoint working", "user_id": user.id, "message_count": count}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
