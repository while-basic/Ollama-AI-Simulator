# ----------------------------------------------------------------------------
#  File:        simulation_loop.py
#  Project:     Celaya Solutions Ollama Simulator
#  Created by:  Celaya Solutions, 2025
#  Author:      Christopher Celaya <chris@celayasolutions.com>
#  Description: Implementation of the main simulation loop
#  Version:     1.0.0
#  License:     MIT (SPDX-Identifier: MIT)
#  Last Update: June 22, 2025
# ----------------------------------------------------------------------------

import time
import yaml
import signal
import select
import threading
import sys
from loguru import logger
from pathlib import Path
from datetime import datetime

from ..agents.mother import MotherLLM
from ..agents.baby import BabyLLM
from ..agents.evaluator import Evaluator
from ..memory.hebbian_store import HebbianMemoryStore
from ..memory.memory_writer import MemoryWriter
from ..memory.dream_engine import DreamEngine
from ..curriculum.lesson_generator import LessonGenerator
from ..curriculum.milestones import MilestoneTracker
from .context_manager import ContextManager
from .logger import SimulationLogger
import ollama

class SimulationLoop:
    """
    Main simulation loop that orchestrates the interaction between Mother and Baby LLMs.
    Manages the day/night cycle and overall simulation flow.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the simulation loop.
        
        Args:
            config_path: Path to the settings.yaml file
        """
        self.config_path = config_path or Path(__file__).parent.parent / "config" / "settings.yaml"
        self.config = self._load_config()
        self.running = False
        self.paused = False
        self.day = 0
        self.interaction_count = 0
        self.dream_cycle_count = 0
        
        # Initialize components
        self._initialize_components()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        logger.info("Simulation loop initialized")
    
    def _load_config(self):
        """Load configuration from the settings.yaml file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Error loading config: {e}")
            return {
                "models": {"mother": "llama3.2:latest", "baby": "llama3.2:1b"},
                "paths": {"obsidian_vault": "./data/obsidian_vault"},
                "learning": {
                    "learning_rate": 0.05,
                    "memory_retention": 0.85,
                    "dream_cycle_interval": 10,
                    "reinforcement_strength": 0.7
                },
                "simulation": {
                    "max_interactions_per_day": 50,
                    "days_to_simulate": 30,
                    "verbose_logging": True
                }
            }
    
    def _initialize_components(self):
        """Initialize all simulation components."""
        # Get model names from config
        mother_model = self.config["models"]["mother"]
        baby_model = self.config["models"]["baby"]
        
        # Initialize memory components
        self.memory_store = HebbianMemoryStore()
        self.memory_writer = MemoryWriter(
            memory_store=self.memory_store,
            embedding_model=baby_model,
            obsidian_path=Path(self.config["paths"]["obsidian_vault"])
        )
        
        # Initialize agents
        self.mother = MotherLLM(model_name=mother_model)
        self.baby = BabyLLM(model_name=baby_model)
        self.evaluator = Evaluator(model_name=mother_model)
        
        # Initialize curriculum components
        self.lesson_generator = LessonGenerator()
        self.milestone_tracker = MilestoneTracker()
        
        # Initialize dream engine
        self.dream_engine = DreamEngine(
            memory_store=self.memory_store,
            mother_model=mother_model,
            baby_model=baby_model
        )
        
        # Initialize context manager
        self.context_manager = ContextManager(memory_store=self.memory_store)
        
        # Initialize logger
        self.simulation_logger = SimulationLogger(
            log_path=Path(self.config["paths"]["logs"]) if "logs" in self.config["paths"] else None,
            verbose=self.config["simulation"].get("verbose_logging", True)
        )
    
    def start(self):
        """Start the simulation loop."""
        import time  # Ensure time is available in this scope
        
        self.running = True
        self.paused = False
        self.day = 0
        self.interaction_count = 0
        self.dream_cycle_count = 0
        
        max_days = self.config["simulation"]["days_to_simulate"]
        
        logger.info(f"Starting simulation for {max_days} days")
        
        print("\n" + "="*80)
        print(f"🧠 OLLAMA SIMULATOR - Mother and Baby LLM Interaction")
        print(f"📚 Simulating {max_days} days of learning")
        print(f"🤖 Mother Model: {self.config['models']['mother']}")
        print(f"👶 Baby Model: {self.config['models']['baby']}")
        print("="*80)
        print("\n💡 Type 'help' for a list of available commands\n")
        
        # Run the actual simulation
        for day in range(1, max_days + 1):
            self.day = day
            print(f"\n\n{'='*40} DAY {day} {'='*40}\n")
            
            # Run the day cycle with actual interactions
            self._run_day_cycle()
            
            # Check for user commands after each day cycle
            print("\n💬 Enter a command (or press Enter to continue): ", end="", flush=True)
            try:
                # Use a non-blocking input with timeout
                import select
                import sys
                
                # Wait for 5 seconds for user input
                i, o, e = select.select([sys.stdin], [], [], 5)
                
                if i:
                    command = sys.stdin.readline().strip()
                    if command:
                        # Handle the command
                        should_continue = self.handle_interactive_command(command)
                        if not should_continue:
                            break
                        
                        # If paused, wait for resume command
                        while self.paused:
                            print("\n💬 Enter a command: ", end="", flush=True)
                            command = input().strip()
                            should_continue = self.handle_interactive_command(command)
                            if not should_continue:
                                break
                        
                        if not should_continue:
                            break
            except Exception as e:
                # Fallback for environments where select doesn't work with stdin
                logger.warning(f"Interactive input error: {e}. Using standard input.")
                try:
                    import threading
                    import time
                    
                    command = ""
                    input_received = False
                    
                    def get_input():
                        nonlocal command, input_received
                        command = input()
                        input_received = True
                    
                    # Start input thread
                    input_thread = threading.Thread(target=get_input)
                    input_thread.daemon = True
                    input_thread.start()
                    
                    # Wait for 5 seconds or until input is received
                    start_time = time.time()
                    while time.time() - start_time < 5 and not input_received:
                        time.sleep(0.1)
                    
                    if input_received and command:
                        # Handle the command
                        should_continue = self.handle_interactive_command(command)
                        if not should_continue:
                            break
                        
                        # If paused, wait for resume command
                        while self.paused:
                            print("\n💬 Enter a command: ", end="", flush=True)
                            command = input().strip()
                            should_continue = self.handle_interactive_command(command)
                            if not should_continue:
                                break
                        
                        if not should_continue:
                            break
                except Exception as e2:
                    logger.error(f"Failed to get interactive input: {e2}")
            
            # Check if we should stop
            if not self.running:
                break
                
            # Run the night cycle for consolidation and dreaming
            print(f"\n{'-'*30} NIGHT CYCLE {'-'*30}\n")
            self._run_night_cycle()
            
            # Save states at the end of each day
            self._save_states()
            
            logger.info(f"Day {day} completed")
            print(f"\n{'='*40} END OF DAY {day} {'='*40}\n")
            
            # Check if we should stop
            if not self.running:
                break
            
            # Check for user commands after each day
            print("\n💬 Enter a command (or press Enter to continue to next day): ", end="", flush=True)
            command = input().strip()
            if command:
                should_continue = self.handle_interactive_command(command)
                if not should_continue:
                    break
                
                # If paused, wait for resume command
                while self.paused:
                    print("\n💬 Enter a command: ", end="", flush=True)
                    command = input().strip()
                    should_continue = self.handle_interactive_command(command)
                    if not should_continue:
                        break
                
                if not should_continue:
                    break
        
        logger.info("Simulation completed")
        print("\n" + "="*80)
        print("🎓 SIMULATION COMPLETED")
        print("="*80 + "\n")
        
        # Enter interactive mode at the end of simulation
        print("💬 Simulation has ended. You can still interact with Mother and Baby LLMs.")
        print("💬 Enter 'exit' to quit the program.")
        
        while True:
            print("\n💬 Enter a command: ", end="", flush=True)
            command = input().strip()
            if not command or command.lower() in ["exit", "quit", "q"]:
                break
            self.handle_interactive_command(command)
    
    def pause(self):
        """Pause the simulation."""
        self.paused = True
        logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation."""
        self.paused = False
        logger.info("Simulation resumed")
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        logger.info("Simulation stopped")
        
        # Save states before stopping
        self._save_states()
        print("\n💾 SAVING STATES: Mother, Baby, and Memory data saved")
    
    def _run_day_cycle(self):
        """Run a day cycle of interactions between Mother and Baby."""
        import time  # Ensure time is available in this scope
        
        max_interactions = self.config["simulation"]["max_interactions_per_day"]
        dream_cycle_interval = self.config["learning"]["dream_cycle_interval"]
        
        # Get baby's current state
        baby_state = self.baby.get_current_state()
        
        # Track interactions for this day
        day_interactions = 0
        
        # Check if there's a last lesson to resume from
        last_lesson = self.mother.get_last_lesson()
        if last_lesson:
            logger.info(f"Resuming from last lesson: {last_lesson.get('topic', 'Unknown')}")
            print(f"\n📚 RESUMING FROM LAST LESSON: {last_lesson.get('topic', 'Unknown')}")
        
        while day_interactions < max_interactions and self.running and not self.paused:
            # Select next topic based on baby's state
            topic = self.lesson_generator.select_next_topic(baby_state)
            
            # Generate lesson with progressive difficulty
            # The mother will automatically adjust difficulty based on baby's performance
            lesson = self.lesson_generator.generate_lesson(
                topic, 
                difficulty_modifier=self.mother.difficulty_level
            )
            
            # Get context for the lesson
            context = self.context_manager.get_context_for_lesson(lesson["content"], baby_state)
            
            print(f"\n{'-'*30} LESSON {day_interactions+1}: {topic} {'-'*30}\n")
            print(f"📊 DIFFICULTY: {self.mother.difficulty_level:.1f}/1.0")
            
            # Generate lesson content from Mother with streaming
            print("👩‍🏫 MOTHER: ", end="", flush=True)
            lesson_content = self.mother.generate_lesson(baby_state, lesson["content"], stream=True)
            print()  # Add a newline after streaming completes
            
            # Log the lesson
            self.simulation_logger.log_lesson(lesson_content, lesson["topic"], self.day, day_interactions)
            
            # Baby responds to the lesson with streaming
            print("\n👶 BABY: ", end="", flush=True)
            baby_response = self.baby.respond_to_lesson(lesson_content, stream=True)
            print()  # Add a newline after streaming completes
            
            # Log the response
            self.simulation_logger.log_response(baby_response, self.day, day_interactions)
            
            # Evaluate the response
            evaluation = self.evaluator.evaluate_response(
                baby_response=baby_response,
                lesson=lesson_content,
                expected_concepts=lesson["expected_concepts"]
            )
            
            # Display evaluation summary
            score = evaluation["overall_score"] if "overall_score" in evaluation else evaluation.get("score", 0.0)
            print(f"\n📊 EVALUATION: Score = {score:.2f}")
            
            # Log the evaluation
            self.simulation_logger.log_evaluation(evaluation, self.day, day_interactions)
            
            # Check for milestones
            achieved_milestones = self.milestone_tracker.check_milestones(
                baby_response=baby_response,
                baby_state=baby_state,
                evaluation=evaluation
            )
            
            # Log and display achieved milestones
            if achieved_milestones:
                print("\n🏆 MILESTONES ACHIEVED:")
                for milestone in achieved_milestones:
                    print(f"   - {milestone}")
                self.simulation_logger.log_milestones(achieved_milestones, self.day, day_interactions)
            
            # Mother provides feedback with streaming
            print("\n👩‍🏫 FEEDBACK: ", end="", flush=True)
            feedback = self.mother.provide_feedback(evaluation, stream=True)
            print()  # Add a newline after streaming completes
            
            # Log the feedback
            self.simulation_logger.log_feedback(feedback, self.day, day_interactions)
            
            # Baby processes the feedback
            self.baby.process_feedback(feedback, score)
            
            # Store memories in Obsidian
            try:
                # Write Mother's lesson to Obsidian
                self.memory_writer.write_mother_log(
                    "lesson",
                    lesson_content,
                    {
                        "topic": lesson["topic"],
                        "day": self.day,
                        "interaction": day_interactions,
                        "difficulty": self.mother.difficulty_level
                    }
                )
                
                # Write Baby's response to Obsidian
                self.memory_writer.write_mother_log(
                    "response",
                    baby_response,
                    {
                        "topic": lesson["topic"],
                        "day": self.day,
                        "interaction": day_interactions,
                        "score": score
                    }
                )
                
                # Write evaluation to Obsidian
                self.memory_writer.write_mother_log(
                    "evaluation",
                    f"Score: {score}\n\n{evaluation.get('comments', '')}", 
                    {
                        "topic": lesson["topic"],
                        "day": self.day,
                        "interaction": day_interactions,
                        "difficulty": self.mother.difficulty_level
                    }
                )
                
                # Write feedback to Obsidian
                self.memory_writer.write_mother_log(
                    "feedback",
                    feedback,
                    {
                        "topic": lesson["topic"],
                        "day": self.day,
                        "interaction": day_interactions,
                        "score": score
                    }
                )
                
                print("\n📝 MEMORY: Writing to Obsidian vault...")
                
                # Store lesson memory in vector database
                lesson_memory_id = self.memory_writer.store_lesson_memory(
                    lesson_content=lesson_content,
                    baby_response=baby_response,
                    evaluation=evaluation
                )
                
                # Store feedback memory in vector database
                feedback_memory_id = self.memory_writer.store_feedback_memory(
                    feedback_content=feedback,
                    score=score
                )
                
                # Create association between lesson and feedback
                self.memory_writer.create_associations_between_memories(
                    [lesson_memory_id, feedback_memory_id]
                )
                
                print("✅ MEMORY: Successfully stored in vector database and Obsidian")
            except Exception as e:
                logger.error(f"Error storing memories: {e}")
                print(f"❌ MEMORY: Error storing memories: {str(e)}")
            
            # Update Mother's record of Baby's progress
            self.mother.update_baby_progress(baby_state, evaluation)
            
            # Update baby state
            baby_state = self.baby.get_current_state()
            
            # Increment counters
            day_interactions += 1
            self.interaction_count += 1
            
            # Check if we need a mini dream cycle
            if day_interactions % dream_cycle_interval == 0:
                self._run_mini_dream_cycle()
            
            # Add a separator between interactions
            print(f"\n{'-'*70}\n")
            
            # Check for user commands after each interaction
            print("\n💬 Enter a command (or press Enter to continue): ", end="", flush=True)
            try:
                # Use a non-blocking input with timeout
                import select
                import sys
                
                # Wait for 3 seconds for user input
                i, o, e = select.select([sys.stdin], [], [], 3)
                
                if i:
                    command = sys.stdin.readline().strip()
                    if command:
                        # Handle the command
                        should_continue = self.handle_interactive_command(command)
                        if not should_continue:
                            self.running = False
                            break
                        
                        # If paused, wait for resume command
                        while self.paused:
                            print("\n💬 Enter a command: ", end="", flush=True)
                            command = input().strip()
                            should_continue = self.handle_interactive_command(command)
                            if not should_continue:
                                self.running = False
                                break
                        
                        if not should_continue:
                            self.running = False
                            break
            except Exception as e:
                # Fallback for environments where select doesn't work with stdin
                logger.warning(f"Interactive input error: {e}. Using standard input.")
                try:
                    import threading
                    import time
                    
                    command = ""
                    input_received = False
                    
                    def get_input():
                        nonlocal command, input_received
                        command = input()
                        input_received = True
                    
                    # Start input thread
                    input_thread = threading.Thread(target=get_input)
                    input_thread.daemon = True
                    input_thread.start()
                    
                    # Wait for 3 seconds or until input is received
                    start_time = time.time()
                    while time.time() - start_time < 3 and not input_received:
                        time.sleep(0.1)
                    
                    if input_received and command:
                        # Handle the command
                        should_continue = self.handle_interactive_command(command)
                        if not should_continue:
                            self.running = False
                            break
                        
                        # If paused, wait for resume command
                        while self.paused:
                            print("\n💬 Enter a command: ", end="", flush=True)
                            command = input().strip()
                            should_continue = self.handle_interactive_command(command)
                            if not should_continue:
                                self.running = False
                                break
                        
                        if not should_continue:
                            self.running = False
                            break
                except Exception as e2:
                    logger.error(f"Failed to get interactive input: {e2}")
            
            # Check if we should stop
            if not self.running:
                break
            
            # Optional pause between interactions
            time.sleep(1)
            
        # Save states at the end of day cycle
        self._save_states()
    
    def _run_night_cycle(self):
        """Run a night cycle for memory consolidation and dreaming."""
        import time  # Ensure time is available in this scope
        
        # Get baby's current state
        baby_state = self.baby.get_current_state()
        
        # Generate dream content
        print("💭 GENERATING DREAM CONTENT...", flush=True)
        dream_content = self.dream_engine.generate_dream(baby_state, stream=True)
        
        # Display dream content with streaming
        print("\n💤 DREAM: ", end="", flush=True)
        # In a real implementation, you would stream this too
        for char in dream_content:
            print(char, end="", flush=True)
            time.sleep(0.01)
        print()  # Add a newline after streaming completes
        
        # Baby processes the dream
        self.baby.process_dream(dream_content)
        
        # Log the dream
        self.simulation_logger.log_dream(dream_content, self.day)
        
        # Write dream to Obsidian
        try:
            print("\n📝 MEMORY: Writing dream to Obsidian vault...")
            
            # Write dream to Obsidian
            self.memory_writer.write_mother_log(
                "dream",
                dream_content,
                {
                    "day": self.day,
                    "dream_cycle": self.dream_cycle_count
                }
            )
            
            # Store dream in vector database
            dream_memory_id = self.memory_writer.store_dream_memory(
                dream_content=dream_content,
                reinforced_concepts=[]  # We'll extract these in a real implementation
            )
            
            print("✅ MEMORY: Dream successfully stored in Obsidian")
            
            # Perform memory consolidation
            print("\n🧠 CONSOLIDATING MEMORIES...")
            consolidation_stats = self.memory_store.consolidate_memories()
            print(f"✅ MEMORY: Consolidated {consolidation_stats.get('strengthened', 0)} connections, pruned {consolidation_stats.get('pruned', 0)} weak connections")
            
            # Generate memory visualization
            print("\n📊 GENERATING MEMORY NETWORK VISUALIZATION...")
            vis_path = self.memory_writer.create_memory_visualization()
            print(f"✅ MEMORY: Visualization created at {vis_path}")
        except Exception as e:
            logger.error(f"Error processing dream memories: {e}")
            print(f"❌ MEMORY: Error processing dream memories: {str(e)}")
        
        # Update dream cycle count
        self.dream_cycle_count += 1
    
    def _run_mini_dream_cycle(self):
        """Run a mini dream cycle for memory reinforcement."""
        import time  # Ensure time is available in this scope
        
        logger.info(f"Running mini dream cycle during day {self.day}")
        
        # Get baby's current state
        baby_state = self.baby.get_current_state()
        
        try:
            # Get recent memories to reinforce
            recent_memories = self.memory_store.get_recent_memories(limit=3)
            
            if not recent_memories:
                print("\n💭 MINI-DREAM CYCLE: No memories to reinforce")
                return
            
            print("\n💭 MINI-DREAM CYCLE: Reinforcing recent memories...")
            
            # Extract content from memories
            memory_contents = [m.get("content", "") for m in recent_memories]
            memory_content = "\n".join([m[:100] + "..." for m in memory_contents])
            
            # Generate mini dream prompt
            prompt = f"""
            Create a short dream sequence that reinforces these recent memories:
            
            {memory_content}
            
            Make it appropriate for the Baby's current development level.
            """
            
            # Generate mini dream content
            mini_dream = ""
            print("\n💤 MINI-DREAM: ", end="", flush=True)
            for chunk in ollama.chat(
                model=self.mother.model_name,
                messages=[
                    {"role": "system", "content": "You are creating a short dream sequence."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    mini_dream += content
            print()  # Add a newline after streaming completes
            
            # Baby processes the mini dream
            self.baby.process_dream(mini_dream)
            
            # Write mini-dream to Obsidian
            self.memory_writer.write_mother_log(
                "mini_dream",
                mini_dream,
                {
                    "day": self.day,
                    "interaction": self.interaction_count
                }
            )
            
            # Log the mini dream
            self.simulation_logger.log_mini_dream(mini_dream, self.day, self.interaction_count)
            
            print("\n✅ MEMORY: Mini-dream processed and stored")
            
        except Exception as e:
            logger.error(f"Error in mini dream cycle: {e}")
            print(f"\n❌ MINI-DREAM ERROR: {str(e)}")
            return
    
    def _save_states(self):
        """Save the states of all components."""
        # Save Baby's state
        self.baby.save_state()
        
        # Save Mother's state
        self.mother.save_state()
        
        # Save memory store
        self.memory_store.save()
        
        # Save milestones
        self.milestone_tracker.save_milestones()
        
        # Generate memory visualization
        try:
            print("\n📊 GENERATING MEMORY NETWORK VISUALIZATION...")
            vis_path = self.memory_writer.create_memory_visualization()
            print(f"✅ MEMORY: Visualization created at {vis_path}")
        except Exception as e:
            logger.error(f"Error generating memory visualization: {e}")
            print(f"❌ MEMORY: Error generating visualization: {str(e)}")
        
        # Generate and save progress report
        progress_report = {
            "day": self.day,
            "interaction_count": self.interaction_count,
            "baby_state": self.baby.get_current_state(),
            "mother_state": self.mother.get_progress_summary(),
            "evaluator_progress": self.evaluator.get_progress_report(),
            "milestones": self.milestone_tracker.get_milestone_summary(),
            "memory_stats": self.memory_store.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.simulation_logger.log_progress_report(progress_report)
        
        # Write progress report to Obsidian
        try:
            self.memory_writer.write_mother_log(
                "progress",
                f"Day {self.day} Progress Report\n\n" +
                f"Baby Vocabulary: {self.baby.state.get('vocabulary_size', 0)} words\n" +
                f"Interactions: {self.interaction_count}\n" +
                f"Current Difficulty: {self.mother.difficulty_level:.1f}/1.0\n" +
                f"Average Score: {self.mother.baby_progress.get('average_score', 0.0):.2f}",
                progress_report
            )
        except Exception as e:
            logger.error(f"Error writing progress report to Obsidian: {e}")
    
    def _cleanup(self):
        """Clean up resources."""
        # Save final states
        self._save_states()
        
        # Log simulation end
        self.simulation_logger.log_simulation_end(
            total_days=self.day,
            total_interactions=self.interaction_count
        )
    
    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signals."""
        logger.info("Interrupt received, stopping simulation...")
        self.stop()
        self._cleanup()
    
    def handle_interactive_command(self, command):
        """
        Handle interactive commands from the user during the simulation.
        
        Args:
            command: The command string from the user
            
        Returns:
            bool: True if the simulation should continue, False if it should stop
        """
        # Strip whitespace and convert to lowercase for easier parsing
        command = command.strip().lower()
        
        # Check for empty command
        if not command:
            return True
            
        # Help command
        if command in ["help", "h", "?"]:
            self._print_help()
            return True
            
        # Status command
        if command in ["status", "s", "info"]:
            self._print_status()
            return True
            
        # Pause/resume commands
        if command in ["pause", "p"]:
            self.pause()
            print("⏸️ Simulation paused. Type 'resume' to continue.")
            return True
            
        if command in ["resume", "r"]:
            self.resume()
            print("▶️ Simulation resumed.")
            return True
            
        # Stop command
        if command in ["stop", "exit", "quit", "q"]:
            self.stop()
            print("🛑 Simulation stopped.")
            return False
            
        # Ask Mother command
        if command.startswith("ask mother ") or command.startswith("ask m "):
            question = command[11:] if command.startswith("ask mother ") else command[6:]
            if question:
                baby_state = self.baby.get_current_state()
                self.mother.answer_user_question(question, baby_state, stream=True)
            else:
                print("❓ Please provide a question to ask the Mother LLM.")
            return True
            
        # Ask Baby command
        if command.startswith("ask baby ") or command.startswith("ask b "):
            question = command[9:] if command.startswith("ask baby ") else command[6:]
            if question:
                self.baby.answer_user_question(question, stream=True)
            else:
                print("❓ Please provide a question to ask the Baby LLM.")
            return True
            
        # Skip to next day
        if command in ["next day", "nextday", "nd"]:
            print("⏭️ Skipping to the next day...")
            return True
            
        # Unknown command
        print(f"❓ Unknown command: '{command}'. Type 'help' to see available commands.")
        return True
        
    def _print_help(self):
        """Print help information about available commands."""
        print("\n" + "="*80)
        print("🔍 AVAILABLE COMMANDS:")
        print("="*80)
        print("  help, h, ?              - Show this help message")
        print("  status, s, info         - Show current simulation status")
        print("  pause, p                - Pause the simulation")
        print("  resume, r               - Resume the simulation")
        print("  stop, exit, quit, q     - Stop the simulation")
        print("  ask mother <question>   - Ask a question to the Mother LLM")
        print("  ask m <question>        - Short form to ask the Mother")
        print("  ask baby <question>     - Ask a question to the Baby LLM")
        print("  ask b <question>        - Short form to ask the Baby")
        print("  next day, nd            - Skip to the next day")
        print("="*80 + "\n")
        
    def _print_status(self):
        """Print current status of the simulation."""
        baby_state = self.baby.get_current_state()
        mother_progress = self.mother.get_progress_summary()
        
        print("\n" + "="*80)
        print("📊 SIMULATION STATUS:")
        print("="*80)
        print(f"  Current Day: {self.day}")
        print(f"  Interactions: {self.interaction_count}")
        print(f"  Baby Vocabulary: {baby_state.get('vocabulary_size', 0)} words")
        print(f"  Baby Age: {baby_state.get('age_days', 0)} days")
        print(f"  Current Difficulty: {self.mother.difficulty_level:.1f}/1.0")
        print(f"  Average Score: {mother_progress.get('average_score', 0.0):.2f}")
        print("="*80 + "\n") 