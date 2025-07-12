# AI Therapist Agent

An advanced AI therapy system that uses multiple therapeutic modalities and ensemble methods to provide adaptive, personalized therapy responses.

## 🌟 Features

- **Multi-Modal Therapy**: Implements 5 evidence-based therapy approaches:
  - Cognitive Behavioral Therapy (CBT)
  - Person-Centered/Empathetic Therapy
  - Solution-Focused Brief Therapy (SFBT)
  - Psychoanalytic/Psychodynamic Therapy
  - Mindfulness-Based Therapy

- **Intelligent Routing**: AI-powered system that selects the most appropriate therapy modality based on patient input
- **Ensemble Approach**: Combines multiple therapy responses for optimal therapeutic outcomes
- **Patient Simulation**: 100+ realistic patient scenarios for testing and evaluation
- **Comparative Analysis**: Benchmarks ensemble vs. baseline approaches

## 🚀 Quick Start

### 1. Setup
```bash
# Clone and navigate to the project
cd ai-therapist

# Run setup script to check dependencies
python setup.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
1. Get an API key from [Together.ai](https://together.ai)
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```bash
   TOGETHER_API_KEY=your_actual_api_key_here
   ```

### 4. Run the Agent
```bash
python agent_v1.py
```

## 📁 Project Structure

```
ai-therapist/
├── agent_v1.py          # Main therapy agent implementation
├── setup.py             # Setup and dependency checker
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Your environment variables (create from .example)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## ⚙️ Configuration

All configuration can be done through environment variables in the `.env` file:

### Required
- `TOGETHER_API_KEY`: Your Together.ai API key

### Optional
- `PATIENT_MODEL`: Model for patient simulation (default: deepseek-ai/DeepSeek-V3)
- `MAX_CONTEXT_LENGTH`: Maximum context length (default: 2048)
- `THERAPIST_MAX_TOKENS`: Max tokens for therapist responses (default: 256)
- `PATIENT_MAX_TOKENS`: Max tokens for patient responses (default: 128)
- `HISTORY_KEEP`: Number of conversation turns to keep (default: 12)
- `SESSION_TURNS`: Number of turns per therapy session (default: 8)

## 🧠 How It Works

1. **Patient Input**: The system receives patient messages/scenarios
2. **Routing**: AI router analyzes input and selects appropriate therapy modalities
3. **Multi-Modal Response**: Multiple therapist agents generate responses using different approaches
4. **Aggregation**: Responses are combined using ensemble methods
5. **Selection**: Best response is selected based on automated evaluation
6. **Evaluation**: Patient simulation provides feedback and ratings

## 📊 Evaluation

The system includes comprehensive evaluation through:
- Patient satisfaction ratings (1-10 scale)
- Before/after therapy assessment
- Comparative analysis between ensemble and baseline approaches
- 100+ diverse patient scenarios covering various mental health presentations

## 🔬 Research Applications

This system is designed for:
- Studying AI therapy effectiveness
- Comparing different therapeutic approaches
- Evaluating ensemble vs. single-model therapy
- Understanding optimal therapy modality selection
- Developing adaptive therapy systems

## 🛠️ Development

### Adding New Therapy Modalities
1. Add the modality to `THERAPISTS` dictionary in `agent_v1.py`
2. Update the `BRAIN_PROMPT` to include selection criteria
3. Test with various patient scenarios

### Customizing Patient Scenarios
Patient scenarios are defined in the `PATIENT_SCENARIOS` list. Each scenario includes:
- Patient background and context
- Specific mental health presentation
- Behavioral patterns and symptoms

## 📚 Therapeutic Approaches

### CBT (Cognitive Behavioral Therapy)
- Focuses on thought patterns and behaviors
- Evidence-based techniques for anxiety, depression
- Structured, goal-oriented approach

### Person-Centered Therapy
- Emphasizes empathy and unconditional positive regard
- Client-led therapy focusing on emotional validation
- Based on Carl Rogers' humanistic approach

### Solution-Focused Brief Therapy (SFBT)
- Goal-oriented, future-focused
- Emphasizes strengths and existing resources
- Brief intervention model

### Psychoanalytic/Psychodynamic
- Explores unconscious patterns and early experiences
- Insight-oriented therapy
- Relationship and attachment focus

### Mindfulness-Based Therapy
- Present-moment awareness and acceptance
- Body-based interventions
- Stress reduction and emotional regulation

## ⚠️ Important Notes

- This system is for research and educational purposes
- Not intended as a replacement for human therapy
- Simulated therapy scenarios for testing AI approaches
- Always consult qualified mental health professionals for real therapeutic needs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure ethical use of AI therapy technologies.

## 🆘 Support

If you encounter issues:
1. Run `python setup.py` to check your setup
2. Verify your API key is correctly set in `.env`
3. Check that all dependencies are installed
4. Review the console output for error messages

For questions about the therapeutic approaches or research applications, please refer to the academic literature on each modality.