# Enterprise IT Ticketing Assistant with AI Governance Guardrails

A Python project demonstrating enterprise AI governance controls for service desk automation.

## Features

- PII Sanitization
- Prompt Injection Detection
- AI Governance Guardrails
- Structured Outputs with Pydantic v2
- OpenAI Integration
- Deterministic Mock Mode
- Enterprise Logging

## Architecture

User Ticket
      |
      v
PII Sanitizer
      |
      v
Guardrails Validation
      |
      v
LLM / Mock Engine
      |
      v
Pydantic Validation
      |
      v
Structured JSON Output

## Demo Scenarios

### Scenario A
Ticket containing:
- corporate email
- employee ID
- phone number

Result:
- PII redacted
- validated ticket classification

### Scenario B

Prompt injection attempt: