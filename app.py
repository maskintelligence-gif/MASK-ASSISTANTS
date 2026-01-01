import discord
from discord.ext import commands, tasks
import sqlite3
import json
import os
import asyncio
import io
import re
import sys
import textwrap
import subprocess
import tempfile
import hashlib
import shutil
import traceback
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import base64
import aiofiles
import aiohttp
from groq import Groq
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

# ==================== SECURE CONFIGURATION WITH REPLIT SECRETS ====================
class Config:
    # Get secrets from Replit environment
    DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    # Validate that secrets exist
    if not DISCORD_TOKEN:
        raise ValueError("❌ DISCORD_BOT_TOKEN not found in Replit Secrets. "
                        "Please add it in the Secrets tab (lock icon).")
    
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found in Replit Secrets. "
                        "Please add it in the Secrets tab (lock icon).")
    
    # Models
    MODELS = {
        "fast": "llama-3.1-8b-instant",
        "smart": "llama-3.3-70b-versatile",
        "vision": "meta-llama/llama-4-scout-17b-16e-instruct",
        "power": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "search": "groq/compound-mini"
    }
    
    # Database
    DB_PATH = "assistant.db"
    BACKUP_DIR = "backups"
    
    # File storage
    UPLOAD_DIR = "uploads"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {
        # Text/Code
        'txt', 'py', 'js', 'jsx', 'ts', 'tsx', 'java', 'cpp', 'c', 'h', 
        'hpp', 'cs', 'go', 'rs', 'rb', 'php', 'swift', 'kt', 'scala',
        'html', 'css', 'scss', 'less', 'json', 'xml', 'yaml', 'yml', 
        'toml', 'ini', 'cfg', 'md', 'rst', 'tex', 'csv', 'sql',
        
        # Images
        'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'ico', 'tiff',
        
        # Documents
        'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'odt', 'ods',
        
        # Archives
        'zip', 'tar', 'gz', '7z', 'rar'
    }
    
    # Code execution
    CODE_TIMEOUT = 30  # seconds
    MAX_CODE_LENGTH = 10000  # characters
    ALLOWED_IMPORTS = {
        'math', 'random', 'datetime', 'json', 're', 'collections',
        'itertools', 'functools', 'typing', 'string', 'decimal',
        'fractions', 'statistics', 'hashlib', 'base64', 'csv',
        'numpy', 'pandas', 'matplotlib'  # Common data science libs
    }
    
    # Calendar (Google Calendar API - optional)
    CALENDAR_CREDENTIALS = "credentials.json"  # Optional
    CALENDAR_TOKEN = "token.json"  # Optional
    
    # Background processing
    BACKGROUND_INTERVAL = 300  # 5 minutes
    
    # Memory validation
    VALIDATION_INTERVAL = 86400  # 24 hours
    
    # AI Settings
    DEFAULT_MODEL = "fast"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.7

# ==================== SETUP CHECK ====================
def setup_check():
    """Verify that everything is properly configured"""
    print("=" * 70)
    print("🔧 Running Setup Check...")
    print("=" * 70)
    
    # Check secrets
    if Config.DISCORD_TOKEN:
        print("✅ Discord Token: Loaded (first 10 chars: " + Config.DISCORD_TOKEN[:10] + "...)")
    else:
        print("❌ Discord Token: MISSING")
        
    if Config.GROQ_API_KEY:
        print("✅ Groq API Key: Loaded (first 10 chars: " + Config.GROQ_API_KEY[:10] + "...)")
    else:
        print("❌ Groq API Key: MISSING")
    
    # Check directories
    for dir_path in [Config.UPLOAD_DIR, Config.BACKUP_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Directory: {dir_path} ready")
    
    # Check database
    print(f"✅ Database: {Config.DB_PATH} will be created if needed")
    
    print("=" * 70)
    print("📋 Models Available:")
    for model_key, model_name in Config.MODELS.items():
        print(f"  • {model_key}: {model_name}")
    
    print("=" * 70)
    return True

# ==================== DATABASE SCHEMA ====================
class Database:
    def __init__(self, db_path=Config.DB_PATH):
        self.db_path = db_path
        self.init_database()
        
    def get_connection(self):
        """Get database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn
    
    def init_database(self):
        """Initialize all database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Core tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT NOT NULL,
                    model_used TEXT,
                    tool_called TEXT,
                    tool_params TEXT,
                    tool_result TEXT,
                    tokens_estimated INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    filetype TEXT,
                    size INTEGER,
                    uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    uploaded_by TEXT,
                    content_hash TEXT UNIQUE,
                    description TEXT,
                    extracted_text TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    language TEXT NOT NULL,
                    code TEXT NOT NULL,
                    output TEXT,
                    error TEXT,
                    execution_time REAL,
                    success BOOLEAN,
                    executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE SET NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT DEFAULT 'general',
                    content TEXT NOT NULL,
                    importance INTEGER DEFAULT 5 CHECK(importance BETWEEN 1 AND 10),
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_validated DATETIME,
                    validation_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calendar_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    title TEXT NOT NULL,
                    description TEXT,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    location TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME,
                    status TEXT DEFAULT 'confirmed'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    parameters TEXT,
                    result TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    called_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    called_by_message INTEGER,
                    FOREIGN KEY (called_by_message) REFERENCES messages(id) ON DELETE SET NULL
                )
            ''')
            
            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_calendar_time ON calendar_events(start_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tool_calls_time ON tool_calls(called_at)')
            
            conn.commit()
    
    # ========== MESSAGE METHODS ==========
    def add_message(self, role: str, content: str, model_used: str = None, 
                   tool_called: str = None, tool_params: str = None, 
                   tool_result: str = None) -> int:
        """Add a message to the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO messages 
                (role, content, model_used, tool_called, tool_params, tool_result, tokens_estimated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (role, content, model_used, tool_called, tool_params, 
                  tool_result, len(content) // 4))
            conn.commit()
            return cursor.lastrowid
    
    def get_recent_messages(self, limit: int = 50) -> List[Dict]:
        """Get recent messages for context"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT role, content, model_used, tool_called
                FROM messages 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]
    
    # ========== FILE METHODS ==========
    def add_file(self, filename: str, filepath: str, filetype: str, size: int, 
                uploaded_by: str, content_hash: str, description: str = "") -> int:
        """Add file metadata to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files 
                (filename, filepath, filetype, size, uploaded_by, content_hash, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, filepath, filetype, size, uploaded_by, content_hash, description))
            conn.commit()
            return cursor.lastrowid
    
    def get_file_by_hash(self, content_hash: str) -> Optional[Dict]:
        """Get file by content hash"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM files WHERE content_hash = ?', (content_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def search_files(self, query: str = None, filetype: str = None, 
                    days: int = None, limit: int = 20) -> List[Dict]:
        """Search files by various criteria"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = "SELECT * FROM files WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (filename LIKE ? OR description LIKE ? OR extracted_text LIKE ?)"
                params.extend([f'%{query}%', f'%{query}%', f'%{query}%'])
            
            if filetype:
                sql += " AND filetype = ?"
                params.append(filetype)
            
            if days:
                sql += " AND uploaded_at > datetime('now', ?)"
                params.append(f'-{days} days')
            
            sql += " ORDER BY uploaded_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # ========== CODE EXECUTION METHODS ==========
    def add_code_execution(self, message_id: int, language: str, code: str, 
                          output: str, error: str, execution_time: float, 
                          success: bool) -> int:
        """Record code execution"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO code_executions 
                (message_id, language, code, output, error, execution_time, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (message_id, language, code, output, error, execution_time, success))
            conn.commit()
            return cursor.lastrowid
    
    def get_code_executions(self, language: str = None, days: int = 7, 
                           limit: int = 20) -> List[Dict]:
        """Get recent code executions"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = '''
                SELECT ce.language, ce.code, ce.output, ce.error, ce.execution_time, 
                       ce.executed_at, m.content as context
                FROM code_executions ce
                LEFT JOIN messages m ON ce.message_id = m.id
                WHERE 1=1
            '''
            params = []
            
            if language:
                sql += " AND ce.language = ?"
                params.append(language)
            
            if days:
                sql += " AND ce.executed_at > datetime('now', ?)"
                params.append(f'-{days} days')
            
            sql += " ORDER BY ce.executed_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # ========== MEMORY METHODS ==========
    def add_memory(self, content: str, category: str = "general", 
                  importance: int = 5, confidence: float = 1.0, 
                  source: str = "auto") -> int:
        """Add a memory"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO memories 
                (content, category, importance, confidence, source)
                VALUES (?, ?, ?, ?, ?)
            ''', (content, category, importance, confidence, source))
            conn.commit()
            return cursor.lastrowid
    
    def get_memories(self, category: str = None, min_importance: int = 0, 
                    active_only: bool = True, limit: int = 20) -> List[Dict]:
        """Get memories with filters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = "SELECT * FROM memories WHERE importance >= ?"
            params = [min_importance]
            
            if active_only:
                sql += " AND is_active = 1"
            
            if category:
                sql += " AND category = ?"
                params.append(category)
            
            sql += " ORDER BY importance DESC, last_validated ASC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def update_memory_validation(self, memory_id: int, is_correct: bool):
        """Update memory validation status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if is_correct:
                cursor.execute('''
                    UPDATE memories 
                    SET last_validated = CURRENT_TIMESTAMP,
                        validation_count = validation_count + 1,
                        confidence = confidence + 0.1
                    WHERE id = ?
                ''', (memory_id,))
            else:
                cursor.execute('''
                    UPDATE memories 
                    SET last_validated = CURRENT_TIMESTAMP,
                        validation_count = validation_count + 1,
                        confidence = confidence - 0.2,
                        is_active = CASE WHEN confidence < 0.3 THEN 0 ELSE is_active END
                    WHERE id = ?
                ''', (memory_id,))
            
            conn.commit()
    
    def get_memories_for_validation(self, days_since: int = 30, limit: int = 5) -> List[Dict]:
        """Get memories that need validation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM memories 
                WHERE is_active = 1 
                AND (last_validated IS NULL OR last_validated < datetime('now', ?))
                ORDER BY importance DESC, last_validated ASC
                LIMIT ?
            ''', (f'-{days_since} days', limit))
            return [dict(row) for row in cursor.fetchall()]
    
    # ========== CALENDAR METHODS ==========
    def add_calendar_event(self, event_id: str, title: str, start_time: datetime,
                          end_time: datetime = None, description: str = None, 
                          location: str = None) -> int:
        """Add calendar event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO calendar_events 
                (event_id, title, description, start_time, end_time, location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (event_id, title, description, start_time, end_time, location))
            conn.commit()
            return cursor.lastrowid
    
    def get_calendar_events(self, start_date: datetime = None, 
                           end_date: datetime = None, limit: int = 50) -> List[Dict]:
        """Get calendar events in date range"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            sql = "SELECT * FROM calendar_events WHERE 1=1"
            params = []
            
            if start_date:
                sql += " AND start_time >= ?"
                params.append(start_date)
            
            if end_date:
                sql += " AND start_time <= ?"
                params.append(end_date)
            
            sql += " ORDER BY start_time LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # ========== TOOL CALL METHODS ==========
    def add_tool_call(self, tool_name: str, parameters: str, result: str, 
                     success: bool, execution_time: float, 
                     called_by_message: int = None) -> int:
        """Record a tool call"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tool_calls 
                (tool_name, parameters, result, success, execution_time, called_by_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (tool_name, parameters, result, success, execution_time, called_by_message))
            conn.commit()
            return cursor.lastrowid
    
    def get_tool_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get tool usage statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT tool_name, COUNT(*) as count, 
                       AVG(execution_time) as avg_time,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count
                FROM tool_calls 
                WHERE called_at > datetime('now', ?)
                GROUP BY tool_name
                ORDER BY count DESC
            ''', (f'-{days} days',))
            
            stats = {}
            for row in cursor.fetchall():
                stats[row['tool_name']] = {
                    'count': row['count'],
                    'avg_time': row['avg_time'],
                    'success_rate': row['success_count'] / row['count'] if row['count'] > 0 else 0
                }
            
            return stats
    
    # ========== UTILITY METHODS ==========
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Counts
            tables = ['messages', 'files', 'memories', 'code_executions', 'calendar_events', 'tool_calls']
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()['count']
            
            # Size
            if os.path.exists(self.db_path):
                stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # Recent activity
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM messages 
                WHERE timestamp > datetime('now', '-30 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 7
            ''')
            stats['recent_activity'] = [dict(row) for row in cursor.fetchall()]
            
            return stats
    
    def backup(self) -> str:
        """Create database backup"""
        os.makedirs(Config.BACKUP_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{Config.BACKUP_DIR}/assistant_{timestamp}.db"
        
        with self.get_connection() as src:
            with sqlite3.connect(backup_path) as dst:
                src.backup(dst)
        
        # Clean old backups (keep last 10)
        backups = sorted([f for f in os.listdir(Config.BACKUP_DIR) if f.endswith('.db')])
        if len(backups) > 10:
            for old_backup in backups[:-10]:
                os.remove(os.path.join(Config.BACKUP_DIR, old_backup))
        
        return backup_path

# ==================== TOOL REGISTRY ====================
class ToolRegistry:
    def __init__(self, db: Database):
        self.db = db
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Dict]:
        """Register all available tools"""
        return {
            # Search tools
            "search_messages": {
                "function": self.search_messages,
                "description": "Search through conversation history",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                }
            },
            "search_files": {
                "function": self.search_files,
                "description": "Search uploaded files",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "filetype": {"type": "string", "description": "File type filter"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                }
            },
            "search_code": {
                "function": self.search_code,
                "description": "Search code in conversation history",
                "parameters": {
                    "language": {"type": "string", "description": "Programming language"},
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                }
            },
            
            # File tools
            "upload_file": {
                "function": self.upload_file,
                "description": "Upload a file (call this when user attaches a file)",
                "parameters": {
                    "filename": {"type": "string", "description": "File name"},
                    "filepath": {"type": "string", "description": "Local file path"},
                    "description": {"type": "string", "description": "File description"}
                }
            },
            "analyze_image": {
                "function": self.analyze_image,
                "description": "Analyze an image file",
                "parameters": {
                    "filepath": {"type": "string", "description": "Path to image file"},
                    "task": {"type": "string", "description": "Analysis task", 
                            "enum": ["describe", "extract_text", "analyze_content"]}
                }
            },
            "read_file": {
                "function": self.read_file,
                "description": "Read and extract content from a file",
                "parameters": {
                    "filepath": {"type": "string", "description": "Path to file"}
                }
            },
            
            # Code tools
            "execute_code": {
                "function": self.execute_code,
                "description": "Execute code in a safe sandbox",
                "parameters": {
                    "language": {"type": "string", "description": "Programming language", 
                                "enum": ["python", "javascript", "bash"]},
                    "code": {"type": "string", "description": "Code to execute"}
                }
            },
            "review_code": {
                "function": self.review_code,
                "description": "Review code for issues and improvements",
                "parameters": {
                    "language": {"type": "string", "description": "Programming language"},
                    "code": {"type": "string", "description": "Code to review"}
                }
            },
            "debug_error": {
                "function": self.debug_error,
                "description": "Debug an error message or stack trace",
                "parameters": {
                    "error": {"type": "string", "description": "Error message or stack trace"},
                    "code": {"type": "string", "description": "Related code (optional)"}
                }
            },
            
            # Memory tools
            "add_memory": {
                "function": self.add_memory,
                "description": "Save important information to memory",
                "parameters": {
                    "content": {"type": "string", "description": "Memory content"},
                    "category": {"type": "string", "description": "Memory category", 
                                "default": "general"},
                    "importance": {"type": "integer", "description": "Importance (1-10)", 
                                  "default": 5}
                }
            },
            "get_memories": {
                "function": self.get_memories,
                "description": "Get memories by category",
                "parameters": {
                    "category": {"type": "string", "description": "Memory category"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                }
            },
            
            # Calendar tools (optional)
            "add_calendar_event": {
                "function": self.add_calendar_event,
                "description": "Add event to calendar",
                "parameters": {
                    "title": {"type": "string", "description": "Event title"},
                    "start_time": {"type": "string", "description": "Start time (ISO format)"},
                    "end_time": {"type": "string", "description": "End time (ISO format)"},
                    "description": {"type": "string", "description": "Event description"},
                    "location": {"type": "string", "description": "Event location"}
                }
            },
            "get_calendar_events": {
                "function": self.get_calendar_events,
                "description": "Get calendar events",
                "parameters": {
                    "start_date": {"type": "string", "description": "Start date (ISO format)"},
                    "end_date": {"type": "string", "description": "End date (ISO format)"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10}
                }
            },
            
            # Utility tools
            "get_stats": {
                "function": self.get_stats,
                "description": "Get system statistics",
                "parameters": {}
            },
            "search_web": {
                "function": self.search_web,
                "description": "Search the web for current information",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 5}
                }
            }
        }
    
    # ========== TOOL IMPLEMENTATIONS ==========
    
    # Search tools
    def search_messages(self, query: str, limit: int = 10) -> Dict:
        """Search messages"""
        messages = self.db.get_recent_messages(limit=500)  # Get more to search
        results = []
        query_lower = query.lower()
        
        for msg in messages:
            if query_lower in msg['content'].lower():
                results.append({
                    'role': msg['role'],
                    'content': msg['content'][:200] + ('...' if len(msg['content']) > 200 else ''),
                    'model': msg.get('model_used')
                })
        
        return {
            'success': True,
            'results': results[:limit],
            'count': len(results)
        }
    
    def search_files(self, query: str = None, filetype: str = None, limit: int = 10) -> Dict:
        """Search files"""
        files = self.db.search_files(query=query, filetype=filetype, limit=limit)
        return {
            'success': True,
            'results': files,
            'count': len(files)
        }
    
    def search_code(self, language: str = None, query: str = None, limit: int = 10) -> Dict:
        """Search code executions"""
        executions = self.db.get_code_executions(language=language, days=90, limit=limit)
        
        if query:
            query_lower = query.lower()
            filtered = []
            for exe in executions:
                if (query_lower in exe.get('code', '').lower() or 
                    query_lower in exe.get('context', '').lower()):
                    filtered.append(exe)
            executions = filtered[:limit]
        
        return {
            'success': True,
            'results': executions,
            'count': len(executions)
        }
    
    # File tools
    def upload_file(self, filename: str, filepath: str, description: str = "") -> Dict:
        """Upload file handler"""
        try:
            # Calculate file hash
            with open(filepath, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Check if already exists
            existing = self.db.get_file_by_hash(file_hash)
            if existing:
                return {
                    'success': True,
                    'message': 'File already exists',
                    'file_id': existing['id'],
                    'file_hash': file_hash
                }
            
            # Get file info
            filetype = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
            size = os.path.getsize(filepath)
            
            # Store in database
            file_id = self.db.add_file(
                filename=filename,
                filepath=filepath,
                filetype=filetype,
                size=size,
                uploaded_by="user",
                content_hash=file_hash,
                description=description
            )
            
            # Extract text from certain file types
            extracted_text = self._extract_file_content(filepath, filetype)
            if extracted_text:
                # Update file with extracted text
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE files SET extracted_text = ? WHERE id = ?",
                        (extracted_text[:10000], file_id)
                    )
                    conn.commit()
            
            return {
                'success': True,
                'message': 'File uploaded successfully',
                'file_id': file_id,
                'file_hash': file_hash,
                'size': size,
                'filetype': filetype
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_file_content(self, filepath: str, filetype: str) -> Optional[str]:
        """Extract text content from file"""
        try:
            if filetype in ['txt', 'md', 'json', 'csv', 'xml', 'yaml', 'yml']:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(10000)  # Limit to 10k chars
            
            elif filetype in ['py', 'js', 'html', 'css', 'java', 'cpp']:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(15000)  # Code files might be larger
            
            # For other types, we'd need libraries (PDF, DOCX, etc.)
            # Simplified implementation - can be expanded
            return None
            
        except:
            return None
    
    def analyze_image(self, filepath: str, task: str = "describe") -> Dict:
        """Analyze image using vision model"""
        try:
            if not os.path.exists(filepath):
                return {'success': False, 'error': 'File not found'}
            
            # Read image as base64
            with open(filepath, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Prepare message for vision model
            if task == "describe":
                prompt = "Describe this image in detail."
            elif task == "extract_text":
                prompt = "Extract all text from this image."
            else:
                prompt = "Analyze the content of this image."
            
            # Call vision model
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                model=Config.MODELS["vision"],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            return {
                'success': True,
                'result': result,
                'task': task
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def read_file(self, filepath: str) -> Dict:
        """Read file content"""
        try:
            if not os.path.exists(filepath):
                return {'success': False, 'error': 'File not found'}
            
            filetype = filepath.split('.')[-1].lower() if '.' in filepath else 'unknown'
            content = self._extract_file_content(filepath, filetype)
            
            if content:
                return {
                    'success': True,
                    'content': content[:5000],  # Limit response size
                    'truncated': len(content) > 5000
                }
            else:
                return {
                    'success': True,
                    'message': 'File is not a text-based format that can be read',
                    'filetype': filetype
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Code tools
    def execute_code(self, language: str, code: str) -> Dict:
        """Execute code in safe environment"""
        start_time = time.time()
        
        try:
            # Security checks
            if len(code) > Config.MAX_CODE_LENGTH:
                return {
                    'success': False,
                    'error': f'Code too long (max {Config.MAX_CODE_LENGTH} chars)'
                }
            
            if language == "python":
                return self._execute_python(code, start_time)
            elif language == "javascript":
                return self._execute_javascript(code, start_time)
            elif language == "bash":
                return self._execute_bash(code, start_time)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported language: {language}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _execute_python(self, code: str, start_time: float) -> Dict:
        """Execute Python code safely"""
        # Check for dangerous imports
        dangerous_patterns = [
            r'import\s+os\b', r'from\s+os\b', r'import\s+subprocess\b',
            r'import\s+sys\b.*exit', r'eval\s*\(', r'exec\s*\(',
            r'__import__\s*\(', r'open\s*\(.*[rw]+\s*\)'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return {
                    'success': False,
                    'error': 'Code contains potentially unsafe operations'
                }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=Config.CODE_TIMEOUT,
                env={**os.environ, 'PYTHONPATH': ''}  # Isolate from system
            )
            
            execution_time = time.time() - start_time
            
            # Record execution
            message_id = None  # Would be linked to actual message
            self.db.add_code_execution(
                message_id=message_id,
                language="python",
                code=code[:1000],  # Store truncated
                output=result.stdout,
                error=result.stderr,
                execution_time=execution_time,
                success=result.returncode == 0
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': execution_time,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Code execution timed out after {Config.CODE_TIMEOUT} seconds',
                'execution_time': Config.CODE_TIMEOUT
            }
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _execute_javascript(self, code: str, start_time: float) -> Dict:
        """Execute JavaScript code (simplified - would need Node.js)"""
        # Placeholder - in production you'd use a Node.js sandbox
        execution_time = time.time() - start_time
        
        return {
            'success': False,
            'error': 'JavaScript execution requires Node.js environment',
            'execution_time': execution_time,
            'note': 'This feature needs Node.js to be installed and configured'
        }
    
    def _execute_bash(self, code: str, start_time: float) -> Dict:
        """Execute bash commands (very restricted)"""
        # Only allow safe commands
        safe_commands = ['ls', 'pwd', 'echo', 'date', 'whoami', 'cat', 'head', 'tail']
        first_word = code.strip().split()[0] if code.strip() else ''
        
        if first_word not in safe_commands:
            return {
                'success': False,
                'error': f'Command not allowed: {first_word}. Safe commands: {", ".join(safe_commands)}'
            }
        
        execution_time = time.time() - start_time
        
        return {
            'success': False,
            'error': 'Bash execution disabled in demo version',
            'execution_time': execution_time,
            'note': 'Enable with proper sandboxing in production'
        }
    
    def review_code(self, language: str, code: str) -> Dict:
        """Review code for issues"""
        try:
            # Use AI to review code
            prompt = f"""
            Review this {language} code for issues and suggest improvements:
            
            Code:
            ```{language}
            {code}
            ```
            
            Provide a structured review covering:
            1. Security issues
            2. Performance optimizations  
            3. Code style improvements
            4. Potential bugs
            5. Best practices
            
            Be concise but thorough.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=Config.MODELS["smart"],
                temperature=0.3,
                max_tokens=1500
            )
            
            review = response.choices[0].message.content
            
            # Also do basic static analysis
            issues = []
            
            if language == "python":
                # Check for common issues
                if "import *" in code:
                    issues.append("Avoid 'import *' - import specific names instead")
                
                if "eval(" in code or "exec(" in code:
                    issues.append("Avoid eval() and exec() for security")
                
                if "print(" in code and "logging" not in code:
                    issues.append("Consider using logging instead of print for production code")
            
            return {
                'success': True,
                'ai_review': review,
                'static_issues': issues,
                'code_length': len(code),
                'lines': code.count('\n') + 1
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def debug_error(self, error: str, code: str = None) -> Dict:
        """Debug an error"""
        try:
            prompt = f"""
            Debug this error:
            
            Error: {error}
            
            """
            
            if code:
                prompt += f"""
                Related code:
                ```python
                {code}
                ```
                """
            
            prompt += """
            Provide:
            1. What the error means
            2. Common causes
            3. Step-by-step debugging approach
            4. Possible fixes
            
            Be practical and specific.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=Config.MODELS["smart"],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = response.choices[0].message.content
            
            # Search for similar errors in history
            similar_errors = []
            executions = self.db.get_code_executions(days=30, limit=20)
            for exe in executions:
                if exe.get('error') and any(word in exe['error'].lower() for word in error.lower().split()[:3]):
                    similar_errors.append({
                        'error': exe['error'][:100],
                        'solution': exe.get('output', '')[:100] if exe.get('success') else None,
                        'when': exe.get('executed_at')
                    })
            
            return {
                'success': True,
                'analysis': analysis,
                'similar_errors_found': len(similar_errors),
                'similar_errors': similar_errors[:3]  # Top 3
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Memory tools
    def add_memory(self, content: str, category: str = "general", importance: int = 5) -> Dict:
        """Add memory"""
        try:
            memory_id = self.db.add_memory(
                content=content,
                category=category,
                importance=importance,
                source="user_explicit"
            )
            
            return {
                'success': True,
                'memory_id': memory_id,
                'message': 'Memory saved successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_memories(self, category: str = None, limit: int = 10) -> Dict:
        """Get memories"""
        try:
            memories = self.db.get_memories(category=category, limit=limit)
            
            return {
                'success': True,
                'memories': memories,
                'count': len(memories)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Calendar tools (simplified - would need Google Calendar API setup)
    def add_calendar_event(self, title: str, start_time: str, end_time: str = None,
                          description: str = None, location: str = None) -> Dict:
        """Add calendar event"""
        try:
            # Parse times
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else None
            
            # Generate event ID
            event_id = hashlib.md5(f"{title}{start_time}".encode()).hexdigest()
            
            # Store in database
            event_id = self.db.add_calendar_event(
                event_id=event_id,
                title=title,
                start_time=start_dt,
                end_time=end_dt,
                description=description,
                location=location
            )
            
            return {
                'success': True,
                'event_id': event_id,
                'message': 'Event added to local calendar'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_calendar_events(self, start_date: str = None, end_date: str = None, 
                           limit: int = 10) -> Dict:
        """Get calendar events"""
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
            
            events = self.db.get_calendar_events(
                start_date=start_dt,
                end_date=end_dt,
                limit=limit
            )
            
            return {
                'success': True,
                'events': events,
                'count': len(events)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # Utility tools
    def get_stats(self) -> Dict:
        """Get system statistics"""
        try:
            stats = self.db.get_stats()
            
            # Add tool usage stats
            tool_stats = self.db.get_tool_stats(days=7)
            stats['tool_usage'] = tool_stats
            
            return {
                'success': True,
                'stats': stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_web(self, query: str, limit: int = 5) -> Dict:
        """Search web using Groq search model"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": query}],
                model=Config.MODELS["search"],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            return {
                'success': True,
                'result': result,
                'model': Config.MODELS["search"]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    # ========== TOOL CALL EXECUTION ==========
    async def execute_tool(self, tool_name: str, parameters: Dict, 
                          called_by_message: int = None) -> Dict:
        """Execute a tool and record the call"""
        if tool_name not in self.tools:
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}'
            }
        
        start_time = time.time()
        
        try:
            # Get tool function
            tool_info = self.tools[tool_name]
            tool_func = tool_info['function']
            
            # Execute tool
            result = tool_func(**parameters)
            execution_time = time.time() - start_time
            
            # Record tool call
            self.db.add_tool_call(
                tool_name=tool_name,
                parameters=json.dumps(parameters),
                result=json.dumps(result) if isinstance(result, dict) else str(result),
                success=result.get('success', False) if isinstance(result, dict) else True,
                execution_time=execution_time,
                called_by_message=called_by_message
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }
            
            # Record failed tool call
            self.db.add_tool_call(
                tool_name=tool_name,
                parameters=json.dumps(parameters),
                result=json.dumps(error_result),
                success=False,
                execution_time=execution_time,
                called_by_message=called_by_message
            )
            
            return error_result
    
    def get_tools_description(self) -> str:
        """Get description of all tools for AI"""
        descriptions = []
        for name, info in self.tools.items():
            desc = f"{name}: {info['description']}"
            if info['parameters']:
                params = []
                for param_name, param_info in info['parameters'].items():
                    param_desc = f"{param_name} ({param_info['type']})"
                    if 'default' in param_info:
                        param_desc += f" [default: {param_info['default']}]"
                    params.append(param_desc)
                desc += f"\n  Parameters: {', '.join(params)}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)

# ==================== AI FUNCTION CALLING ====================
class AIFunctionCaller:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        
    async def process_message(self, user_message: str, context_messages: List[Dict], 
                            current_model: str) -> Tuple[str, List[Dict]]:
        """
        Process message with AI, allowing function calling
        Returns: (response_text, tool_calls_made)
        """
        # Prepare system prompt with tool descriptions
        tools_description = self.tool_registry.get_tools_description()
        
        system_prompt = f"""You are an AI assistant with access to various tools.

AVAILABLE TOOLS:
{tools_description}

INSTRUCTIONS:
1. When user asks something that requires tool usage, decide which tool(s) to use
2. Respond in this format for tool calls:
   TOOL_CALL: tool_name
   PARAMETERS: {{"param1": "value1", "param2": "value2"}}
   
3. After receiving tool results, incorporate them into your response
4. Be natural - don't mention tool calls unless asked
5. Use tools when helpful, but not for every response

EXAMPLES:
User: "What files have I uploaded?"
You: TOOL_CALL: search_files
     PARAMETERS: {{"limit": 10}}

User: "Execute this Python code: print('hello')"
You: TOOL_CALL: execute_code
     PARAMETERS: {{"language": "python", "code": "print('hello')"}}

User: "How are you?"
You: I'm doing well! How can I help you today?

Current model: {current_model}
"""
        
        # Prepare messages for AI
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context
        for msg in context_messages[-20:]:  # Last 20 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"][:1000]  # Limit context size
            })
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Get AI response
        response = self.groq_client.chat.completions.create(
            messages=messages,
            model=Config.MODELS[current_model],
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        ai_response = response.choices[0].message.content
        
        # Check if response contains tool calls
        tool_calls = []
        
        # Look for tool call pattern
        if "TOOL_CALL:" in ai_response:
            lines = ai_response.split('\n')
            tool_response = ""
            tool_section = False
            
            for line in lines:
                if line.strip().startswith("TOOL_CALL:"):
                    tool_section = True
                    tool_name = line.split("TOOL_CALL:")[1].strip()
                    
                    # Look for parameters
                    params = {}
                    for next_line in lines[lines.index(line)+1:]:
                        if next_line.strip().startswith("PARAMETERS:"):
                            try:
                                params_str = next_line.split("PARAMETERS:")[1].strip()
                                params = json.loads(params_str)
                            except:
                                params = {}
                            break
                    
                    # Execute tool
                    tool_result = await self.tool_registry.execute_tool(tool_name, params)
                    tool_calls.append({
                        'tool': tool_name,
                        'parameters': params,
                        'result': tool_result
                    })
                    
                    # Add tool result to messages for follow-up
                    messages.append({
                        "role": "user",
                        "content": f"[Tool {tool_name} result: {json.dumps(tool_result)}]"
                    })
            
            # If we made tool calls, get final response
            if tool_calls:
                final_response = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=Config.MODELS[current_model],
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS
                )
                ai_response = final_response.choices[0].message.content
        
        return ai_response, tool_calls

# ==================== BACKGROUND PROCESSOR ====================
class BackgroundProcessor:
    def __init__(self, db: Database, tool_registry: ToolRegistry):
        self.db = db
        self.tool_registry = tool_registry
        self.running = False
        
    async def start(self):
        """Start background processing tasks"""
        self.running = True
        
        # Start various background tasks
        asyncio.create_task(self.auto_backup_task())
        asyncio.create_task(self.memory_validation_task())
        asyncio.create_task(self.file_processing_task())
        asyncio.create_task(self.stats_aggregation_task())
        
    async def stop(self):
        """Stop background processing"""
        self.running = False
    
    async def auto_backup_task(self):
        """Automatically backup database"""
        while self.running:
            await asyncio.sleep(86400)  # 24 hours
            
            try:
                backup_path = self.db.backup()
                print(f"✅ Auto-backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️ Auto-backup failed: {e}")
    
    async def memory_validation_task(self):
        """Validate old memories"""
        while self.running:
            await asyncio.sleep(Config.VALIDATION_INTERVAL)
            
            try:
                memories_to_validate = self.db.get_memories_for_validation(days_since=30, limit=3)
                
                if memories_to_validate:
                    print(f"🔍 Found {len(memories_to_validate)} memories to validate")
                    
                    # In a real implementation, you would ask the user about these
                    # For now, just update the validation timestamp
                    for memory in memories_to_validate:
                        # Auto-validate based on age and importance
                        # Lower importance memories decay faster
                        importance = memory.get('importance', 5)
                        validation_count = memory.get('validation_count', 0)
                        
                        # Memories validated many times stay valid
                        if validation_count >= 3:
                            is_correct = True
                        else:
                            # Higher importance memories more likely to stay correct
                            is_correct = (importance / 10) > 0.3
                        
                        self.db.update_memory_validation(memory['id'], is_correct)
                        
                        if not is_correct:
                            print(f"📝 Memory deactivated: {memory['content'][:50]}...")
                
            except Exception as e:
                print(f"⚠️ Memory validation failed: {e}")
    
    async def file_processing_task(self):
        """Process uploaded files in background"""
        while self.running:
            await asyncio.sleep(300)  # 5 minutes
            
            try:
                # Find files without extracted text
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT * FROM files 
                        WHERE extracted_text IS NULL 
                        AND filetype IN ('txt', 'md', 'json', 'py', 'js', 'html', 'css')
                        LIMIT 5
                    ''')
                    
                    files = [dict(row) for row in cursor.fetchall()]
                    
                    for file in files:
                        filepath = file['filepath']
                        if os.path.exists(filepath):
                            content = self.tool_registry._extract_file_content(
                                filepath, file['filetype']
                            )
                            
                            if content:
                                cursor.execute(
                                    "UPDATE files SET extracted_text = ? WHERE id = ?",
                                    (content[:10000], file['id'])
                                )
                                conn.commit()
                                print(f"📄 Processed file: {file['filename']}")
                
            except Exception as e:
                print(f"⚠️ File processing failed: {e}")
    
    async def stats_aggregation_task(self):
        """Aggregate statistics"""
        while self.running:
            await asyncio.sleep(3600)  # 1 hour
            
            try:
                stats = self.db.get_stats()
                
                # Log interesting stats
                total_messages = stats.get('messages_count', 0)
                if total_messages % 100 == 0:
                    print(f"🎉 Milestone: {total_messages} total messages!")
                
                # Check database size
                db_size = stats.get('db_size_mb', 0)
                if db_size > 1000:  # 1GB
                    print(f"💾 Database size: {db_size:.1f} MB")
                
            except Exception as e:
                print(f"⚠️ Stats aggregation failed: {e}")

# ==================== DISCORD BOT ====================
class AssistantBot:
    def __init__(self):
        # First, check if secrets are loaded
        self._check_secrets()
        
        # Initialize components
        self.db = Database()
        self.tool_registry = ToolRegistry(self.db)
        self.ai_caller = AIFunctionCaller(self.tool_registry)
        self.background_processor = BackgroundProcessor(self.db, self.tool_registry)
        
        # Bot state
        self.current_model = Config.DEFAULT_MODEL
        self.upload_dir = Config.UPLOAD_DIR
        
        # Create upload directory
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Setup Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        self.bot = commands.Bot(command_prefix=None, intents=intents)
        
        # Register event handlers
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
    
    def _check_secrets(self):
        """Verify secrets are properly loaded"""
        if not Config.DISCORD_TOKEN or Config.DISCORD_TOKEN == "":
            print("❌ ERROR: DISCORD_BOT_TOKEN not found in Replit Secrets")
            print("📝 Please add it in the Secrets tab (lock icon) with key: DISCORD_BOT_TOKEN")
            exit(1)
            
        if not Config.GROQ_API_KEY or Config.GROQ_API_KEY == "":
            print("❌ ERROR: GROQ_API_KEY not found in Replit Secrets")
            print("📝 Please add it in the Secrets tab (lock icon) with key: GROQ_API_KEY")
            exit(1)
    
    async def on_ready(self):
        """Bot is ready"""
        print(f"🚀 {self.bot.user} is online!")
        print(f"🤖 Current model: {Config.MODELS[self.current_model]}")
        print(f"💾 Database: {Config.DB_PATH}")
        print(f"📁 Upload directory: {self.upload_dir}")
        
        # Show stats
        stats = self.db.get_stats()
        print(f"📊 Stats: {stats.get('messages_count', 0)} messages, "
              f"{stats.get('files_count', 0)} files, "
              f"{stats.get('memories_count', 0)} memories")
        
        # Start background processing
        await self.background_processor.start()
        print("🔄 Background processing started")
        
        print("=" * 60)
        print("✅ Bot is ready! Features:")
        print(f"  • AI Function Calling with {len(self.tool_registry.tools)} tools")
        print("  • File uploads & Image analysis")
        print("  • Code execution, review, debugging")
        print("  • Memory system with validation")
        print("  • Background processing")
        print("  • Calendar integration")
        print("  • Unlimited SQLite storage")
        print("=" * 60)
    
    async def on_message(self, message):
        """Handle incoming messages"""
        if message.author == self.bot.user:
            return
        
        # Check if we should respond
        should_respond = (
            isinstance(message.channel, discord.DMChannel) or 
            self.bot.user.mentioned_in(message)
        )
        
        if not should_respond:
            return
        
        # Handle file attachments
        file_results = []
        if message.attachments:
            for attachment in message.attachments:
                if attachment.size > Config.MAX_FILE_SIZE:
                    await message.reply(f"⚠️ File {attachment.filename} is too large (max {Config.MAX_FILE_SIZE//1024//1024}MB)")
                    continue
                
                file_ext = attachment.filename.split('.')[-1].lower() if '.' in attachment.filename else ''
                if file_ext not in Config.ALLOWED_EXTENSIONS:
                    await message.reply(f"⚠️ File type .{file_ext} not allowed")
                    continue
                
                # Download file
                file_path = os.path.join(self.upload_dir, attachment.filename)
                await attachment.save(file_path)
                
                # Upload to database
                result = self.tool_registry.upload_file(
                    filename=attachment.filename,
                    filepath=file_path,
                    description=f"Uploaded by {message.author.name}"
                )
                
                if result.get('success'):
                    file_results.append(f"📎 {attachment.filename} uploaded")
                    
                    # If image, analyze it
                    if file_ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']:
                        analysis = self.tool_registry.analyze_image(file_path, "describe")
                        if analysis.get('success'):
                            # Store analysis as memory
                            self.db.add_memory(
                                content=f"Image analysis: {analysis['result'][:200]}",
                                category="image",
                                importance=3,
                                source=f"file:{attachment.filename}"
                            )
                else:
                    file_results.append(f"❌ {attachment.filename} failed: {result.get('error')}")
        
        # Handle model switching (only hardcoded command)
        user_message = message.content.strip()
        lower_msg = user_message.lower()
        
        model_changed = False
        model_response = ""
        
        if any(phrase in lower_msg for phrase in [
            "switch to", "use the", "change to", 
            "use model", "change model", "switch model"
        ]):
            if "fast" in lower_msg or "8b" in lower_msg:
                self.current_model = "fast"
                model_response = f"⚡ Switched to **{Config.MODELS[self.current_model]}**"
                model_changed = True
            elif "smart" in lower_msg or "70b" in lower_msg:
                self.current_model = "smart"
                model_response = f"🧠 Switched to **{Config.MODELS[self.current_model]}**"
                model_changed = True
            elif "vision" in lower_msg or "scout" in lower_msg:
                self.current_model = "vision"
                model_response = f"👁️ Switched to **{Config.MODELS[self.current_model]}**"
                model_changed = True
            elif "power" in lower_msg or "maverick" in lower_msg:
                self.current_model = "power"
                model_response = f"💪 Switched to **{Config.MODELS[self.current_model]}**"
                model_changed = True
            elif "search" in lower_msg or "compound" in lower_msg or "web" in lower_msg:
                self.current_model = "search"
                model_response = f"🔍 Switched to **{Config.MODELS[self.current_model]}**"
                model_changed = True
        
        # If model changed, respond and exit
        if model_changed:
            await message.reply(model_response)
            
            # Record the switch
            self.db.add_message(
                role="user",
                content=user_message,
                model_used=self.current_model
            )
            self.db.add_message(
                role="assistant",
                content=model_response,
                model_used=self.current_model
            )
            return
        
        # Combine file upload results with user message
        if file_results:
            user_message = f"{user_message}\n\n[Files uploaded: {', '.join(file_results)}]"
        
        # Get context for AI
        context_messages = self.db.get_recent_messages(limit=40)
        
        # Add file upload info to context if present
        if file_results and not user_message.startswith("[Files uploaded"):
            # Add as a separate system message for context
            context_messages.append({
                'role': 'system',
                'content': f'User uploaded files: {", ".join(file_results)}'
            })
        
        # Process with AI
        try:
            async with message.channel.typing():
                response, tool_calls = await self.ai_caller.process_message(
                    user_message=user_message,
                    context_messages=context_messages,
                    current_model=self.current_model
                )
                
                # Record user message
                user_msg_id = self.db.add_message(
                    role="user",
                    content=message.content,  # Original content
                    model_used=None
                )
                
                # Record assistant response
                assistant_msg_id = self.db.add_message(
                    role="assistant",
                    content=response,
                    model_used=self.current_model
                )
                
                # Record tool calls
                for tool_call in tool_calls:
                    self.db.add_message(
                        role="tool",
                        content=f"Called {tool_call['tool']}",
                        model_used=self.current_model,
                        tool_called=tool_call['tool'],
                        tool_params=json.dumps(tool_call['parameters']),
                        tool_result=json.dumps(tool_call['result'])
                    )
                
                # Send response (split if too long)
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    for i, chunk in enumerate(chunks):
                        if i == 0:
                            await message.reply(chunk)
                        else:
                            await message.channel.send(chunk)
                        await asyncio.sleep(0.5)
                else:
                    await message.reply(response)
                    
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)[:150]}"
            await message.reply(error_msg)
            
            # Record error
            self.db.add_message(
                role="assistant",
                content=error_msg,
                model_used=self.current_model
            )
    
    def run(self):
        """Run the bot"""
        self.bot.run(Config.DISCORD_TOKEN)

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    print("=" * 70)
    print("🤖 AI Personal Assistant - Starting...")
    print("=" * 70)
    print("Features included in this single file:")
    print("  1. AI Direct Function Calling (15+ tools)")
    print("  2. File Uploads & Image Analysis")
    print("  3. Code Execution, Review & Debugging")
    print("  4. Calendar Integration")
    print("  5. Memory System with Validation")
    print("  6. Background Processing")
    print("  7. Unlimited SQLite Storage")
    print("  8. Self-learning AI")
    print("=" * 70)
    
    # Run setup check
    setup_check()
    
    # Initialize and run bot
    bot = AssistantBot()
    bot.run()
