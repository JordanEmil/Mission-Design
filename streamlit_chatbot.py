#!/usr/bin/env python3
"""
Space Mission RAG Chatbot - Streamlit Interface
Web-based interface for querying space mission knowledge base
"""

# SQLite compatibility fix for Streamlit Cloud
import sqlite_fix

import os
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import hashlib

import streamlit as st
from query_pipeline import SpaceMissionQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from database_manager import get_database_manager
from auth_manager import AuthManager


class StreamlitSpaceMissionChatbot:
    """Streamlit-based chatbot for space mission queries"""
    
    def __init__(self, log_dir: str = "./chat_logs"):
        """
        Initialize the Streamlit chatbot interface
        
        Args:
            log_dir: Directory to store chat logs
        """
        self.log_dir = Path(log_dir)
        self.db_manager = get_database_manager()
        self.auth_manager = AuthManager()
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.messages = []
            st.session_state.session_id = self._generate_session_id()
            st.session_state.session_start = datetime.now()
            st.session_state.show_sources = True
            st.session_state.query_count = 0
            st.session_state.query_engine = None
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.show_login = False
            st.session_state.show_login_modal = False
            st.session_state.show_signup_modal = False
            st.session_state.current_page = "chat"
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def _save_message_to_db(self, role: str, content: str, sources: List[Dict] = None):
        """Save a single message to database for logged in users"""
        if st.session_state.current_user:
            self.db_manager.save_chat_message(
                session_id=st.session_state.session_id,
                message_type=role,
                message=content,
                sources=sources,
                user_id=st.session_state.current_user['id']
            )
    
    def _load_user_history(self):
        """Load chat history for logged in users"""
        if st.session_state.current_user and len(st.session_state.messages) == 0:
            # Load recent messages from current session
            history = self.db_manager.get_session_chat_history(st.session_state.session_id)
            for msg in history:
                st.session_state.messages.append({
                    "role": msg["message_type"],
                    "content": msg["message"],
                    "timestamp": msg["created_at"],
                    "sources": msg.get("sources", [])
                })
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Format source documents for display with deduplication"""
        if not sources:
            return ""
            
        # Deduplicate sources based on mission_id
        seen_missions = {}
        unique_sources = []
        
        for source in sources:
            metadata = source.get('metadata', {})
            mission_id = metadata.get('mission_id', '')
            
            if mission_id:
                if mission_id not in seen_missions:
                    seen_missions[mission_id] = source
                    unique_sources.append(source)
                elif source.get('score', 0) > seen_missions[mission_id].get('score', 0):
                    idx = unique_sources.index(seen_missions[mission_id])
                    unique_sources[idx] = source
                    seen_missions[mission_id] = source
            else:
                unique_sources.append(source)
            
            if len(unique_sources) >= 20:
                break
        
        # Sort by score descending
        unique_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        formatted = "\n\n**Sources:**"
        for i, source in enumerate(unique_sources, 1):
            metadata = source.get('metadata', {})
            title = metadata.get('title', 'Unknown Mission')
            url = metadata.get('url', '')
            score = source.get('score', 0)
            
            # Clean up title if needed
            if title and ' - eoPortal' in title:
                title = title.replace(' - eoPortal', '')
            
            # Format with URL if available
            if url:
                if not url.startswith('http'):
                    url = f"https://{url}"
                formatted += f"\n{i}. [{title}]({url}) (relevance: {score:.3f})"
            else:
                formatted += f"\n{i}. **{title}** (relevance: {score:.3f})"
        
        return formatted
    
    def _process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result"""
        try:
            result = st.session_state.query_engine.query(
                query,
                response_mode=ResponseMode.COMPACT,
                return_sources=True,
                verbose=False
            )
            
            # Increment query count
            st.session_state.query_count += 1
            
            return result
            
        except Exception as e:
            return {
                'response': f"Error processing query: {str(e)}",
                'sources': [],
                'metadata': {'response_time': 0}
            }
    
    def render_header(self):
        """Render header with navigation and auth buttons"""
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            # Page navigation
            cols = st.columns(3)
            with cols[0]:
                if st.button("Chat", use_container_width=True):
                    st.session_state.current_page = "chat"
                    st.rerun()
            with cols[1]:
                if st.button("About", use_container_width=True):
                    st.session_state.current_page = "about"
                    st.rerun()
        
        with col2:
            if st.session_state.current_user:
                st.button(f"{st.session_state.current_user['username']}", disabled=True)
            else:
                if st.button("Login", use_container_width=True):
                    st.session_state.show_login_modal = True
                    st.rerun()
        
        with col3:
            if st.session_state.current_user:
                if st.button("Logout", use_container_width=True):
                    st.session_state.authenticated = False
                    st.session_state.current_user = None
                    st.session_state.messages = []
                    st.rerun()
            else:
                if st.button("Sign Up", use_container_width=True):
                    st.session_state.show_signup_modal = True
                    st.rerun()
    
    def render_login_modal(self):
        """Render login modal dialog"""
        @st.dialog("Login")
        def login_dialog():
            with st.form("login_form"):
                username = st.text_input("Username or Email")
                password = st.text_input("Password", type="password")
                col1, col2 = st.columns(2)
                
                with col1:
                    login_button = st.form_submit_button("Login", use_container_width=True)
                with col2:
                    cancel_button = st.form_submit_button("Cancel", use_container_width=True)
                
                if login_button:
                    user = self.db_manager.authenticate_user(username, password)
                    if user:
                        st.session_state.current_user = user
                        st.session_state.authenticated = True
                        st.session_state.show_login_modal = False
                        st.success(f"Welcome back, {user['username']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                
                if cancel_button:
                    st.session_state.show_login_modal = False
                    st.rerun()
        
        if st.session_state.show_login_modal:
            login_dialog()
    
    def render_signup_modal(self):
        """Render signup modal dialog"""
        @st.dialog("Sign Up")
        def signup_dialog():
            with st.form("signup_form"):
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                col1, col2 = st.columns(2)
                
                with col1:
                    signup_button = st.form_submit_button("Sign Up", use_container_width=True)
                with col2:
                    cancel_button = st.form_submit_button("Cancel", use_container_width=True)
                
                if signup_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        user = self.db_manager.create_user(new_username, new_email, new_password)
                        if user:
                            st.session_state.current_user = user
                            st.session_state.authenticated = True
                            st.session_state.show_signup_modal = False
                            st.success("Account created successfully!")
                            st.rerun()
                        else:
                            st.error("Username or email already exists")
                
                if cancel_button:
                    st.session_state.show_signup_modal = False
                    st.rerun()
        
        if st.session_state.show_signup_modal:
            signup_dialog()
    
    def _display_pdf_large(self, pdf_path, caption=""):
        """Display a PDF file in the Streamlit app"""
        try:
            with open(pdf_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'
                st.markdown(pdf_display, unsafe_allow_html=True)
                if caption:
                    st.caption(caption)
        except Exception as e:
            st.error(f"Could not display PDF: {e}")
    
    def render_about_page(self):
        """Render the about page"""
        st.title("About the Space Mission Design Assistant")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Profile picture placeholder
            profile_path = Path(__file__).parent / "profile_picture.jpg"
            if profile_path.exists():
                st.image(str(profile_path), width=200)
            else:
                # Try current directory as fallback
                if os.path.exists("profile_picture.jpg"):
                    st.image("profile_picture.jpg", width=200)
                else:
                    st.info("Add profile_picture.jpg to the project directory")
        
        with col2:
            st.markdown("""
## Emil Ares  
        **Space Mission Design Assistant — RAG for Space Missions**


        I built this assistant as part of my MSc Thesis at Cranfield University to let engineers query the
        eoPortal knowledge base using Retrieval-Augmented Generation (RAG). Before this, I was a Physics undergraduate at the University of Cambridge.
                        
        The assistant combines **LlamaIndex**, **ChromaDB**, and an OpenAI LLM to return grounded, citeable answers about orbits, payloads, buses, and mission patterns.

        **Why trust it?** I ran a full evaluation with **synthetically generated Q/As** (LLM-assisted)
        and **RAGAS** metrics, plus classical IR metrics. The configuration you're using here is based
        on that study's Pareto-optimal trade between answer quality and latency.
        """)

        # Credentials / contact
        st.markdown("""
        **Email:** [eja65@cantab.ac.uk](mailto:eja65@cantab.ac.uk)  
        **LinkedIn:** [Emil Ares](https://www.linkedin.com/in/emil-ares/)
        ---
        """)

        # --- Evaluation Summary (from IRP) ---
        st.markdown("""
        ### Evaluation Summary
        - **Retriever quality:** Mean Average Precision **≈ 0.95**; high MRR — relevant passages usually rank at the top.  
        - **Answer quality:** LLM-judged **F1 ≈ 0.75** on the synthetic benchmark; strong faithfulness per **RAGAS**.  
        - **Latency:** Median end-to-end response **≈ 30.0 s** on the prototype.  
        - **Selected runtime config:** `top_k=5`, `similarity_threshold=0.1`, `temperature=0.1`, `ResponseMode=COMPACT`.  
        - **Key finding:** Retrieval quality and final answer quality are **strongly correlated**; tightening similarity too much (e.g. 0.5) hurts recall and overall answers.
        """)

        # Inline citations to the IRP & figure doc
        st.caption("See IRP for full methodology, parameter sweeps, and metrics tables. "
                   "Correlation heatmap shown below. "
                   ":contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}")

    st.markdown("---")

    # --- Big correlation heatmap (bridging_9_correlation_heatmap.pdf) ---
    # Resolve probable locations (assets or /mnt/data)
    possible_paths = [
        Path(__file__).parent / "assets" / "bridging_9_correlation_heatmap.pdf",
        Path("assets/bridging_9_correlation_heatmap.pdf"),
        Path("/mnt/data/bridging_9_correlation_heatmap.pdf"),
        Path("bridging_9_correlation_heatmap.pdf"),
    ]
    heatmap_path = next((str(p) for p in possible_paths if p.exists()), None)

    st.subheader("How the system's metrics relate (Correlation Heatmap)")
    if heatmap_path:
        self._display_pdf_large(
            heatmap_path,
            caption="Correlation heatmap across retrieval (Recall@k, MAP, MRR), generation (EM, F1, ROUGE-L, BLEU), and RAGAS metrics."
        )
    else:
        st.warning("Could not locate `bridging_9_correlation_heatmap.pdf`. Place it in `assets/` or `/mnt/data/`.")

    # --- Explain the figure clearly to users (simple first, then technical) ---
    st.markdown("""
    **Intuition:** when the search step finds the right passages, the final answer is better.  
    The heatmap is mostly warm/positive between retrieval metrics (e.g., **MAP**, **MRR**) and
    answer metrics (e.g., **F1**, **ROUGE-L**) — that's exactly what we want to see.

    **What to look at:**
    - **Retrieval ↔ Answer quality:** Strong positive correlations show that improving retrieval
      (better ranking of relevant context) boosts final answer faithfulness and relevance.
    - **Over-strict filters:** A too-high similarity threshold (e.g., 0.7) starves the LLM of context,
      which drops both retrieval and generation scores.
    - **RAGAS signals:** **R-Faith** (faithfulness) and **R-AnsRel** (answer relevance) move with
      classical metrics, confirming that grounded, on-source responses track with good retrieval.

    **Bottom line for users:** this assistant is tuned to return a small set of highly relevant
    passages (k=5) with a moderate similarity threshold (0.1). That balance gave the best quality-per-second in testing.
    """)

    st.markdown("---")

    # --- Short "About the Project/Tech Stack" tails for completeness ---
    st.markdown("""
    ### Project & Stack
    - **Corpus:** ~1,000+ mission pages from ESA **eoPortal** (scraped with a robust, high-throughput pipeline).
    - **Indexing:** Markdown structuring, semantic embeddings, vector search in **ChromaDB**.
    - **Orchestration:** **LlamaIndex** QueryEngine with source grouping & dedup.
    - **Evaluation:** synthetic Q/A generation, RAGAS, classical IR metrics, Pareto analysis (F1 vs latency).
    """)

    # Small tagline to make your expertise explicit to visitors:
    st.info("Built by Emil Ares — **RAG enthusiast**. Synthetic Q/A + **RAGAS** used to tune and validate this assistant.")
    
    def render_sidebar(self):
        """Render the sidebar with settings and information"""
        with st.sidebar:
            st.header("Space Mission Chatbot")
            
            # User info
            if st.session_state.current_user:
                st.subheader("Session Info")
                st.text(f"User: {st.session_state.current_user['username']}")
                st.text(f"Session ID: {st.session_state.session_id}")
                st.text(f"Queries: {st.session_state.query_count}")
            else:
                st.subheader("Session Info")
                st.text("Guest User")
                st.text(f"Session ID: {st.session_state.session_id}")
                st.text(f"Queries: {st.session_state.query_count}")
            
            # Settings
            st.subheader("Settings")
            st.session_state.show_sources = st.checkbox(
                "Show source documents",
                value=st.session_state.show_sources
            )
            
            # Actions
            st.subheader("Actions")
            
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.query_count = 0
                st.rerun()
            
            if st.session_state.current_user:
                if st.button("View History"):
                    history = self.db_manager.get_user_chat_history(
                        st.session_state.current_user['id'], 
                        limit=50
                    )
                    if history:
                        st.subheader("Recent Queries")
                        for msg in history[:10]:
                            if msg['message_type'] == 'user':
                                st.text(f"Q: {msg['message'][:50]}...")
                    else:
                        st.info("No history yet")
            
            # Examples
            st.subheader("Example Questions")
            examples = [
                "What orbit regimes have been used for SAR imaging satellites?",
                "What are typical power requirements for Earth observation CubeSats?",
                "Which missions have used optical imaging payloads?",
                "What are common failure modes in small satellite missions?",
                "Compare antenna designs used in different SAR missions"
            ]
            
            for example in examples:
                if st.button(example, key=f"example_{hash(example)}"):
                    st.session_state.example_query = example
                    st.rerun()
            
            # Stats
            if st.checkbox("Show Engine Stats"):
                if st.session_state.query_engine:
                    stats = st.session_state.query_engine.get_engine_stats()
                    st.json(stats)
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Load user history if logged in and messages are empty
        if st.session_state.current_user and len(st.session_state.messages) == 0:
            self._load_user_history()
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show metadata if available
                if "metadata" in message and message["role"] == "assistant":
                    st.caption(f"Response time: {message['metadata']['response_time']:.2f}s")
        
        # Handle example query if set
        if hasattr(st.session_state, 'example_query'):
            query = st.session_state.example_query
            del st.session_state.example_query
        else:
            # Chat input
            query = st.chat_input(
                "Ask about space missions, orbits, payloads, etc..."
            )
        
        if query:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save user message to DB if logged in
            self._save_message_to_db("user", query)
            
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Process query with loading indicator
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    result = self._process_query(query)
                
                # Display response
                response_text = result['response']
                
                # Add sources if enabled
                sources_data = []
                if st.session_state.show_sources and result.get('sources'):
                    response_text += self._format_sources(result['sources'])
                    sources_data = result['sources']
                
                st.write(response_text)
                
                # Show metadata
                query_time = result['metadata'].get('response_time', 0)
                st.caption(f"Response time: {query_time:.2f}s")
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": result['metadata'],
                    "sources": sources_data
                })
                
                # Save assistant message to DB if logged in
                self._save_message_to_db("assistant", response_text, sources_data)
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Space Mission Design Assistant",
            page_icon="🚀",
            layout="wide"
        )
        
        # Apply custom CSS for better styling
        st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }
        div[data-testid="stHorizontalBlock"] > div:first-child {
            flex-grow: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize query engine if not already done
        if not st.session_state.initialized:
            try:
                # Get API key from secrets
                api_key = st.secrets.get("OPENAI_API_KEY")
                if not api_key:
                    st.error("OpenAI API key not found in Streamlit secrets!")
                    st.info("Please add OPENAI_API_KEY to your Streamlit secrets.")
                    st.stop()
                
                # Set the API key in environment
                os.environ["OPENAI_API_KEY"] = api_key
                
                with st.spinner("Initializing Space Mission Assistant..."):
                    st.session_state.query_engine = SpaceMissionQueryEngine(
                        use_cloud=True,  # Use ChromaDB cloud
                        top_k=5, # Retrieve top 5 documents
                        similarity_threshold=0.1, # Lower threshold for more results
                        temperature=0.1, # Optimal result
                        llm_model="o3"
                    )
                    st.session_state.initialized = True
                    st.session_state.authenticated = True  # Allow immediate access
            except Exception as e:
                st.error(f"Error initializing query engine: {e}")
                st.stop()
        
        # Render header with navigation
        self.render_header()
        
        # Render modals if needed
        self.render_login_modal()
        self.render_signup_modal()
        
        # Render content based on current page
        if st.session_state.current_page == "about":
            self.render_about_page()
        else:
            # Render sidebar
            self.render_sidebar()
            
            # Main content area
            st.title("🚀 Space Mission Design Assistant")
            st.markdown("Ask questions about historical space missions, orbits, payloads, and mission designs.")
            
            # Render chat interface
            self.render_chat_interface()


def main():
    """Main function to run the Streamlit chatbot"""
    # Create and run chatbot
    chatbot = StreamlitSpaceMissionChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()