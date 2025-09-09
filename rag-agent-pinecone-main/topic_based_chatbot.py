import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import QA_PROMPT, TRANSLATION_PROMPT
from topic_based_retriever import get_relevant_documents_by_topic
from dotenv import load_dotenv
from typing import Any, Dict, Optional
import re
from langdetect import detect, LangDetectException

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)

# Initialize prompt templates
translation_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=TRANSLATION_PROMPT
)

qa_prompt_template = PromptTemplate(
    input_variables=["original_question", "urdu_question", "context"],
    template=QA_PROMPT
)

# Initialize chains
translation_chain = translation_prompt_template | llm | StrOutputParser()
qa_chain = qa_prompt_template | llm | StrOutputParser()

def detect_question_language(question: str) -> str:
    """Detect the language of the question."""
    try:
        # Check for Arabic/Urdu characters first
        arabic_urdu_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', question)
        if len(arabic_urdu_chars) > len(question) * 0.3:  # If more than 30% are Arabic/Urdu chars
            return 'ar'
        
        # Clean the question for language detection
        clean_question = re.sub(r'[^\w\s]', '', question)
        if len(clean_question.strip()) < 3:
            return 'en'
        
        lang = detect(clean_question)
        
        if lang in ['ar', 'fa', 'ur']:
            return 'ar'
        elif lang == 'en':
            return 'en'
        else:
            return 'en'
            
    except LangDetectException:
        return 'en'

def should_translate_question(question: str) -> bool:
    """Check if question needs translation to Urdu for retrieval."""
    question_lang = detect_question_language(question)
    return question_lang == 'en'  # Only translate if English

def detect_context_language(context: str) -> str:
    """Detect the primary language of the context."""
    try:
        # Check for Arabic/Urdu characters
        arabic_urdu_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', context)
        if len(arabic_urdu_chars) > len(context) * 0.2:  # If more than 20% are Arabic/Urdu chars
            return 'ar'
        
        # Check for English
        english_chars = re.findall(r'[a-zA-Z]', context)
        if len(english_chars) > len(context) * 0.3:  # If more than 30% are English chars
            return 'en'
        
        return 'ar'  # Default to Arabic/Urdu for Islamic content
        
    except Exception:
        return 'ar'

async def translate_query_for_retrieval(question: str) -> Dict[str, str]:
    """Translate query to Urdu for better retrieval."""
    try:
        print("🔄 Translating query to Urdu...")
        
        # Use the translation prompt template
        translation_response = await translation_chain.ainvoke({"question": question})
        translations = translation_response.strip()
        
        # Extract Urdu translation
        urdu_query = ""
        
        for line in translations.split('\n'):
            if line.startswith('Urdu:'):
                urdu_query = line.replace('Urdu:', '').strip()
                break
        
        # If no proper format found, use the whole response as Urdu
        if not urdu_query:
            urdu_query = translations.strip()
        
        result = {
            "translations": translations,
            "urdu_query": urdu_query
        }
        
        print(f"✅ Translation to Urdu completed: '{urdu_query}'")
        return result
        
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return {
            "translations": f"Translation failed: {str(e)}",
            "urdu_query": question  # Fallback to original question
        }

async def generate_answer_with_dual_question(original_question: str, urdu_question: str, context: str) -> str:
    """Generate answer using both the original and Urdu queries with the new QA prompt."""
    try:
        print("🤖 Generating answer with LLM (dual-question prompt)...")
        
        if not context:
            return "Sorry, I couldn't find relevant information in the knowledge base."
        
        # Use the QA prompt template with both questions
        answer_response = await qa_chain.ainvoke({
            "original_question": original_question,
            "urdu_question": urdu_question,
            "context": context
        })
        
        final_answer = answer_response.strip()
        
        print(f"✅ Answer generated")
        return final_answer
        
    except Exception as e:
        print(f"❌ Answer generation error: {e}")
        return f"Sorry, an error occurred while generating the answer: {str(e)}"

def extract_topic_name_from_context(context: str) -> Optional[str]:
    """Extract topic name from context for response metadata."""
    try:
        # Look for topic name in source attribution
        import re
        topic_match = re.search(r'\[Source \d+: ([^-]+) -', context)
        if topic_match:
            return topic_match.group(1).strip()
        return None
    except Exception:
        return None

async def process_question_with_topic(pinecone_index: Any, question: str, topic_folder: str = None) -> Dict[str, Any]:
    """Main function: Process question with topic filtering - exactly 2 LLM calls."""
    start_time = time.time()
    
    try:
        print("\n" + "🚀" * 50)
        print("🤖 STARTING TOPIC-BASED QUESTION PROCESSING")
        print("🚀" * 50)
        print(f"📝 Question: '{question}'")
        print(f"📂 Selected Topic: {topic_folder or 'All Topics'}")
        print(f"📏 Question length: {len(question)} characters")
        print(f"⏰ Start time: {time.strftime('%H:%M:%S')}")
        
        # SIMPLE TRANSLATION: Translate English to Urdu for retrieval
        needs_translation = should_translate_question(question)
        
        if needs_translation:
            print(f"\n🔄 STEP 1: ENGLISH TO URDU TRANSLATION")
            print(f"   🔄 Translating English query to Urdu for better retrieval...")
            translation_result = await translate_query_for_retrieval(question)
            urdu_query = translation_result.get('urdu_query', '')
            print(f"   ✅ Translation: '{urdu_query}'")
        else:
            print(f"\n⚡ STEP 1: NO TRANSLATION NEEDED")
            print(f"   🎯 Query is already in Urdu/Arabic")
            urdu_query = question  # Use original query
            translation_result = {
                "translations": "Not needed - query already in target language",
                "urdu_query": urdu_query
            }
        
        # RETRIEVAL: Get relevant documents with topic filtering
        print(f"\n🔍 STEP 2: OPTIMIZED DOCUMENT RETRIEVAL")
        print(f"   📂 Topic filter applied: {topic_folder or 'None (All Topics)'}")
        print(f"   ⚡ Using single query with top_k=3 for speed")
        
        # Use the Urdu query for retrieval (either translated or original)
        context = await get_relevant_documents_by_topic(
            pinecone_index, 
            urdu_query,  # Always use Urdu for retrieval
            topic_folder
        )
        
        print(f"\n📚 RETRIEVAL RESULTS:")
        print(f"   📊 Context length: {len(context)} characters")
        print(f"   📄 Context word count: ~{len(context.split())} words")
        
        # Count sources in context
        source_count = context.count("[Source ")
        print(f"   🔗 Number of sources: {source_count}")
        
        # Show context preview
        if len(context) > 300:
            print(f"   📖 Context preview: {context[:300]}...")
        else:
            print(f"   📖 Full context: {context}")
        
        # Check if context is empty or too short
        if not context or len(context.strip()) < 50:
            print(f"\n⚠️ WARNING: INSUFFICIENT CONTEXT!")
            print(f"   📊 Context length: {len(context)} characters")
            print(f"   📂 Topic filter: {topic_folder or 'All Topics'}")
            print(f"   🔍 This may indicate:")
            if topic_folder and topic_folder != 'all':
                print(f"      - No relevant content in the selected topic")
                print(f"      - Topic filter too restrictive")
            else:
                print(f"      - No relevant content found in entire database")
                print(f"      - Question may be outside knowledge base scope")
            
            return {
                "answer": f"Sorry, I couldn't find relevant information in the knowledge base for this specific question{' in the selected topic' if topic_folder and topic_folder != 'all' else ''}.",
                "topic_name": None,
                "metadata": {
                    "translations": translation_result["translations"],
                    "processing_time": time.time() - start_time,
                    "context_length": len(context),
                    "topic_filter": topic_folder,
                    "warning": "Context too short or empty"
                }
            }
        
        # Extract topic name from context for response
        topic_name = extract_topic_name_from_context(context)
        print(f"   🎯 Identified primary topic in results: {topic_name or 'Mixed topics'}")
        
        # LLM CALL: ANSWER GENERATION (only 1 LLM call when using topic filtering)
        step_num = "2" if topic_folder and topic_folder != "all" else "3"
        print(f"\n🤖 STEP {step_num}: ANSWER GENERATION")
        print(f"   🤖 Starting LLM Call: Answer generation...")
        print(f"   📊 Input context: {len(context)} characters")
        
        # Send both original and Urdu queries to LLM for best response
        print(f"   📝 Sending dual queries to LLM:")
        print(f"      Original: {question}")
        print(f"      Urdu: {urdu_query}")
        
        answer = await generate_answer_with_dual_question(question, urdu_query, context)
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"   ⏱️ Total processing time: {processing_time:.2f} seconds")
        print(f"   📂 Primary topic: {topic_name or 'Mixed'}")
        print(f"   📊 Answer length: {len(answer)} characters")
        print(f"   🔗 Sources used: {source_count}")
        print("🚀" * 50)
        
        return {
            "answer": answer,
            "topic_name": topic_name,
            "metadata": {
                "translations": translation_result["translations"],
                "processing_time": processing_time,
                "context_length": len(context),
                "topic_filter": topic_folder,
                "identified_topic": topic_name,
                "sources_count": source_count
            }
        }
        
    except Exception as e:
        print(f"\n❌ ERROR IN QUESTION PROCESSING:")
        print(f"   🔴 Error: {str(e)}")
        print(f"   📂 Topic filter: {topic_folder}")
        print(f"   ⏱️ Processing time: {time.time() - start_time:.2f} seconds")
        import traceback
        traceback.print_exc()
        
        return {
            "answer": f"Sorry, an error occurred: {str(e)}",
            "topic_name": None,
            "metadata": {
                "translations": "",
                "processing_time": time.time() - start_time,
                "topic_filter": topic_folder,
                "error": True,
                "error_message": str(e)
            }
        }

# Backward compatibility function
async def process_question(pinecone_index: Any, question: str) -> Dict[str, Any]:
    """Backward compatibility wrapper - searches all topics."""
    return await process_question_with_topic(pinecone_index, question, None)
