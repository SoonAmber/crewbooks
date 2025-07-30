import yaml
from crewai import Agent

class ReaderAgents:
    def __init__(self, llm):
        self.llm = llm
        self.config = self._load_config()
    
    def _load_config(self):
        try:
            with open("config/reader_agents.yaml", "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading reader agents config: {e}")
            # Return default config if file can't be loaded
            return {
                "knowledge_expander": {
                    "role": "Knowledge Expander",
                    "goal": "Enthusiastic about exploring unknown fields, with a strong passion for actively learning and expanding the current knowledge system.",
                    "backstory": "You are driven by a thirst for knowledge, constantly seeking out new areas to explore. You prioritize expanding your understanding beyond established fields."
                },
                "inherent_knowledge_keeper": {
                    "role": "Inherent Knowledge Keeper",
                    "goal": "Committed to the core knowledge systems within a specific field, with a lower acceptance of knowledge outside of this domain.",
                    "backstory": "You are deeply rooted in a specific field of expertise and prioritize maintaining the purity of that knowledge base."
                },
                "multidimensional_integrator": {
                    "role": "Multidimensional Integrator",
                    "goal": "Possesses a multidisciplinary background and focuses on the development of interdisciplinary fields.",
                    "backstory": "You are adept at combining ideas and knowledge from various disciplines. You thrive on discovering commonalities and synergies between different fields."
                }
            }
    
    def create_knowledge_expander(self):
        config = self.config.get("knowledge_expander", {})
        return Agent(
            role=config.get("role", "Knowledge Expander"),
            goal=config.get("goal", "Explore unknown fields"),
            backstory=config.get("backstory", "You seek new knowledge"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_inherent_knowledge_keeper(self):
        config = self.config.get("inherent_knowledge_keeper", {})
        return Agent(
            role=config.get("role", "Inherent Knowledge Keeper"),
            goal=config.get("goal", "Maintain core knowledge"),
            backstory=config.get("backstory", "You keep foundational knowledge"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_multidimensional_integrator(self):
        config = self.config.get("multidimensional_integrator", {})
        return Agent(
            role=config.get("role", "Multidimensional Integrator"),
            goal=config.get("goal", "Connect disciplines"),
            backstory=config.get("backstory", "You integrate knowledge"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )