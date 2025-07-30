# src/tasks/reader_tasks.py
import yaml
from crewai import Task

class ReaderTasks:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        with open("config/reader_tasks.yaml", "r") as file:
            return yaml.safe_load(file)

    def reader_question(self, agent, topic):
        config = self.config["reader_question"]
        agent_role = agent.role
        return Task(
            description=config["description"].format(topic=topic, agent_role=agent_role),
            expected_output=config["expected_output"].format(agent_role=agent_role),
            agent=agent
        )

    def book_description(self, agent, question):
        config = self.config["book_description"]
        return Task(
            description=f"{config['description']} Based on your question: {question}",
            expected_output=config["expected_output"],
            agent=agent
        )

    def recommendation_evaluation(self, agent, recommendations):
        config = self.config["recommendation_evaluation"]
        return Task(
            description=f"{config['description']} Recommendations to evaluate: {recommendations}",
            expected_output=config["expected_output"],
            agent=agent
        )

    def format_final_requirements(self, agent, expanded_perspective, core_perspective, integrated_perspective):
        """Format the three perspectives into a single set of requirements"""
        return Task(
            description=f"""
            Based on these three perspectives:
            
            1. Knowledge Expander's Perspective:
            {expanded_perspective}
            
            2. Inherent Knowledge Keeper's Perspective:
            {core_perspective}
            
            3. Multidimensional Integrator's Perspective:
            {integrated_perspective}
            
            Synthesize these perspectives into a comprehensive book recommendation requirement.
            Create three distinct requirement descriptions that capture the essence of each perspective
            while ensuring they are clear and actionable for the library staff.
            """,
            expected_output="Three distinct structured requirements for book recommendations",
            agent=agent
        )