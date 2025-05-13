# Ollama Simulator: Mother and Baby LLM

A simulation environment that uses Ollama to create a "Mother" LLM (using llama3.2:latest) and a "Baby" LLM (using llama3.2:1b) to explore how smaller language models can learn from larger ones through guided interactions.

## 🧠 Core Concept

The Ollama Simulator creates a learning environment where:

* **Mother LLM**: A high-capability model (llama3.2:latest) acts as teacher, parent, and guide. Controls curriculum, feedback, and personality shaping.
* **Baby LLM**: A smaller model (llama3.2:1b) that starts with little knowledge and learns through guided prompt completion, reinforcement via scoring and memory updates, and repeated exposure to stimuli from the Mother.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- Obsidian (optional, for knowledge base integration)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ollama-simulator.git
   cd ollama-simulator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure settings:
   - Edit `ollama_simulator/config/settings.yaml` to adjust simulation parameters
   - Edit `ollama_simulator/config/personas.yaml` to adjust Mother LLM personality traits

4. Run the simulator:
   ```bash
   python ollama_simulator.py
   ```

### Command Line Options

```bash
python ollama_simulator.py --help
```

Available options:
- `--config PATH`: Path to config file (default: config/settings.yaml)
- `--mode {cli,web,tui}`: Interface mode (default: cli)
- `--days DAYS`: Number of days to simulate (overrides config)
- `--obsidian PATH`: Path to Obsidian vault for logging
- `--mother MODEL`: Mother LLM model to use
- `--baby MODEL`: Baby LLM model to use
- `--verbose`: Enable verbose logging

## 🔄 Training Loop (Simulated Learning)

The simulation follows a cycle where:

1. **Mother** presents a lesson or prompt
2. **Baby** responds based on its current knowledge
3. **Mother** evaluates and provides feedback
4. Baby's memory is updated based on the interaction
5. During "night" cycles, memories are consolidated through a dream process

## 📚 Growth Stages

The Baby LLM progresses through different developmental stages:

| Stage | Age Range | Capabilities & Traits |
|-------|-----------|------------------------|
| 👶 **Infant** | 0–1 LLM Days | Sound mimicking, emotional tone detection, single-word understanding |
| 🐣 **Toddler** | 2–5 Days | Simple vocab use, basic grammar, emotional response learning |
| 🧒 **Child** | 6–15 Days | Sentence structure, asks questions, learns cause/effect |
| 🧑 **Teenager** | 16–30 Days | Learns ethics, begins abstract thought, sarcasm, emotional instability |
| 👨 **Adult** | 30+ Days | Complex reasoning, goal-setting, memory refinement |
| 🧙 **Elder** | 60+ Days | Compresses memories, mentors younger AIs, creates philosophical logs |

## 🧪 Features

- **Hebbian Learning**: Baby's memory uses a neural-inspired approach where "neurons that fire together, wire together"
- **Dream Engine**: Consolidates memories during night cycles
- **Multiple UIs**: Command line, terminal UI, and web dashboard options
- **Obsidian Integration**: Journals from both Mother and Baby can be saved into Obsidian vaults
- **Personality Traits**: Configure different teaching styles for the Mother LLM

## 📊 Monitoring and Visualization

The simulator includes:
- Real-time learning progress tracking
- Memory visualization
- Milestone achievement logging
- Interaction history

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 