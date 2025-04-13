# Technical Design Document: Technical Interviewer Voice Bot

## 1. System Architecture Overview

This document outlines the technical design for a FAANG-style technical interviewer voice bot using the Cloud-Assisted approach with Model Context Protocol (MCP) integration.

```
┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│                │     │                  │     │                 │     │                  │
│  User Speech   │────▶│  Cloud STT API   │────▶│  MCP Context    │────▶│  Cloud LLM API   │
│                │     │  (Whisper/       │     │  Manager        │     │  (OpenAI/Claude) │
└────────────────┘     │   Deepgram)      │     │                 │     │                  │
                       └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                        │                         │
                                                        │                         │
                                                        ▼                         ▼
┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                │     │                  │     │                 │
│  Audio Output  │◀────│  Sesame CSM      │◀────│  MCP Context    │
│                │     │  (Local)         │     │  Manager        │
└────────────────┘     │                  │     │                 │
                       └──────────────────┘     └─────────────────┘
```

## 2. Core Components

### 2.1 Speech-to-Text (STT) Service
- **Implementation**: Cloud API (Whisper API or Deepgram)
- **Function**: Converts user's spoken answers to text 
- **Key Features**:
  - Accurate transcription of technical programming terms
  - Real-time streaming with low latency
  - Punctuation and sentence segmentation

### 2.2 Model Context Protocol (MCP) Manager
- **Implementation**: Custom Python module following MCP specification
- **Function**: Maintains structured conversation context, code state, and interview progress
- **Key Features**:
  - Standardized schema for technical interview context
  - Efficient token usage optimization
  - State preservation between components

### 2.3 Large Language Model (LLM)
- **Implementation**: Cloud API (OpenAI GPT-4, Anthropic Claude, or Google Gemini)
- **Function**: Processes user responses, analyzes code, generates interviewer questions
- **Key Features**:
  - Sophisticated code analysis and feedback
  - Dynamic interviewing strategy based on user skill level
  - Function calling for structured actions

### 2.4 Text-to-Speech (TTS)
- **Implementation**: Local Sesame CSM running on GPU
- **Function**: Converts LLM responses to natural-sounding speech
- **Key Features**:
  - High-quality voice synthesis for interviewer persona
  - Consistent voice characteristics throughout session
  - Context-aware speech generation

### 2.5 Code Editor Integration
- **Implementation**: WebSocket API for real-time code transfer
- **Function**: Sends user code to LLM for analysis, receives feedback
- **Key Features**:
  - Bi-directional code transfer
  - Syntax highlighting preservation
  - Real-time analysis during coding

## 3. MCP Schema Design

### 3.1 Core MCP Schema Extensions
```json
{
  "interview": {
    "session_id": "string",
    "problem": {
      "title": "string",
      "difficulty": "string",
      "category": "string",
      "description": "string",
      "constraints": ["string"],
      "examples": [{
        "input": "string",
        "output": "string",
        "explanation": "string"
      }]
    },
    "state": {
      "phase": "introduction|problem_statement|clarification|coding|testing|feedback|conclusion",
      "elapsed_time": "number",
      "remaining_time": "number"
    }
  },
  "code": {
    "current_version": "string",
    "language": "string",
    "history": [{
      "timestamp": "string",
      "version": "string",
      "analysis": {
        "complexity_time": "string",
        "complexity_space": "string",
        "issues": [{
          "type": "string",
          "description": "string",
          "line_number": "number"
        }],
        "edge_cases": [{
          "description": "string",
          "handled": "boolean"
        }]
      }
    }]
  },
  "participant": {
    "progress": {
      "problem_understanding": "number",
      "algorithmic_thinking": "number",
      "code_quality": "number",
      "communication": "number",
      "testing_thoroughness": "number"
    },
    "hints_given": [{
      "timestamp": "string",
      "hint": "string",
      "trigger": "string"
    }]
  },
  "interviewer": {
    "persona": {
      "style": "supportive|neutral|challenging",
      "technical_focus": ["string"],
      "question_patterns": ["string"]
    },
    "planned_interventions": [{
      "trigger_condition": "string",
      "intervention_type": "hint|question|clarification|redirect",
      "content": "string"
    }]
  }
}
```

### 3.2 MCP State Transitions
- **Introduction → Problem Statement**: Triggered after initial greeting
- **Problem Statement → Clarification**: Triggered when problem fully described
- **Clarification → Coding**: Triggered when sufficient problem understanding demonstrated
- **Coding → Testing**: Triggered when initial solution submitted
- **Testing → Feedback**: Triggered when testing complete or time threshold reached
- **Feedback → Conclusion**: Triggered when feedback fully delivered

## 4. Speech Synthesis with Sesame CSM

### 4.1 Sesame Integration
- **Model**: CSM-1B with dual-speaker capability
- **Inference**: Local GPU acceleration (CUDA/MPS)
- **Prompt Design**: Technical interviewer persona
- **Context Retention**: Maintain speech characteristics across session

### 4.2 Speaker Configuration
```python
INTERVIEWER_PROMPT = {
    "text": "carefully analyzed the approach and I can see you're using a depth-first search algorithm here. That's interesting. Can you walk me through why you chose this over a breadth-first approach? I'm curious about your reasoning on the time complexity as well.",
    "speaker_id": 0,
    "audio": "interviewer_prompt.wav"
}
```

### 4.3 Audio Processing Pipeline
1. Text from LLM passed through MCP context manager
2. Context-sensitive generation parameters determined
3. Speech generated with consistent voice characteristics
4. Audio streamed to client with minimal latency

## 5. Cloud APIs Integration

### 5.1 STT Service Configuration
```python
stt_config = {
    "provider": "whisper_api",  # Alternative: "deepgram"
    "model": "whisper-1",
    "language": "en",
    "options": {
        "word_timestamps": True,
        "punctuation": True,
        "technical_vocabulary": True
    }
}
```

### 5.2 LLM Service Configuration
```python
llm_config = {
    "provider": "openai",  # Alternatives: "anthropic", "google"
    "model": "gpt-4-turbo",
    "temperature": 0.2,
    "max_tokens": 1024,
    "functions": [
        {
            "name": "analyze_code",
            "description": "Analyzes code for correctness and efficiency",
            "parameters": {
                "code": "string",
                "language": "string"
            }
        },
        {
            "name": "provide_hint",
            "description": "Provides a hint based on current progress",
            "parameters": {
                "difficulty": "string",
                "specific_issue": "string"
            }
        },
        # Additional function definitions
    ]
}
```

## 6. Pipeline Data Flow

### 6.1 User Speech Processing Path
1. User's speech captured via microphone
2. Audio streamed to cloud STT service
3. Transcribed text wrapped in MCP format
4. Contextualized with interview state
5. Passed to LLM with relevant code context

### 6.2 System Response Path
1. LLM generates interviewer response
2. Response packaged in MCP format
3. Context state updated
4. Text passed to Sesame CSM
5. Generated speech streamed to user

### 6.3 Code Analysis Path
1. Code editor content synchronized via WebSocket
2. Code wrapped in MCP format with history
3. Analyzed by LLM for correctness, efficiency, edge cases
4. Analysis results stored in MCP context
5. Feedback incorporated into interviewer responses

## 7. Error Handling & Resilience

### 7.1 STT Error Handling
- Fallback transcription options for technical terms
- Confidence scoring for ambiguous transcriptions
- User confirmation for critical instructions

### 7.2 LLM Error Handling
- Timeout and retry mechanism with exponential backoff
- Response validation against expected schema
- Content filtering for inappropriate responses
- Graceful degradation for API unavailability

### 7.3 TTS Error Handling
- Local model initialization validation
- Fallback to simpler synthesis if context issues occur
- Resource monitoring to prevent GPU memory exhaustion

## 8. Performance Considerations

### 8.1 Latency Targets
- End-to-end response time: <2000ms
- STT processing: <500ms
- LLM processing: <1000ms
- TTS processing: <500ms

### 8.2 Resource Requirements
- GPU: NVIDIA T4 or better for Sesame CSM
- Memory: 16GB system RAM, 16GB GPU memory
- Network: <50ms latency to cloud APIs
- Storage: 10GB for models and context history

### 8.3 Optimization Strategies
- Parallel processing of non-dependent components
- Response caching for common interviewer patterns
- Streaming responses for perceived latency reduction
- Context pruning to minimize token usage

## 9. Security & Privacy

### 9.1 Data Handling
- Ephemeral storage of voice recordings
- Encrypted transmission of all data
- Anonymized session logging
- User consent management

### 9.2 API Security
- Secure API key storage via environment variables
- Rate limiting protection
- IP restriction for production deployment

## 10. Monitoring & Observability

### 10.1 Key Metrics
- Component-level latency measurement
- End-to-end response time
- API call success rates
- Token usage efficiency
- User interruption frequency

### 10.2 Logging Strategy
- Structured JSON logs with correlation IDs
- MCP state transitions logged
- Error conditions with context
- Performance anomaly detection
