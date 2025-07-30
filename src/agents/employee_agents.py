import yaml
from crewai import Agent

class EmployeeAgents:
    def __init__(self, llm, library_tools):
        self.llm = llm
        self.library_tools = library_tools
        self.config = self._load_config()
    
    def _load_config(self):
        try:
            with open("config/employee_agents.yaml", "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading employee agents config: {e}")
            # Return default config if file can't be loaded
            return {
                "demand_assistant": {
                    "role": "Demand Assistant",
                    "goal": "Analyze the given needs, decompose the needs into small needs, and arrange the priorities of the needs.",
                    "backstory": "You are an expert in requirements analysis and knowledge mapping. Your expertise lies in breaking down complex needs into manageable components."
                },
                "retrieval_assistant": {
                    "role": "Retrieval Specialist",
                    "goal": "Search existing resources in csv and other platforms to collect relevant books.",
                    "backstory": "You possess advanced capabilities in sourcing resources across diverse platforms."
                },
                "organization_assistant": {
                    "role": "Organization Specialist",
                    "goal": "To screen and evaluate the resources retrieved based on their authenticity, authority, and relevance.",
                    "backstory": "You possess expertise in organizing and managing data. Your specialization lies in filtering information."
                },
                "collection_assistant": {
                    "role": "Collection Assistant",
                    "goal": "Manage book information database synchronization and add new entries.",
                    "backstory": "You are responsible for managing csv files's physical and digital resources."
                },
                "recommendation_assistant": {
                    "role": "Recommendation Specialist",
                    "goal": "According to demand_assistant, the final list is given according to the list given by collection_assistant.",
                    "backstory": "You excel at understanding the user's evolving needs and matching them with the most relevant resources available in the library."
                }
            }
    
    def create_demand_assistant(self):
        config = self.config.get("demand_assistant", {})
        return Agent(
            role=config.get("role", "Demand Assistant"),
            goal=config.get("goal", "Analyze needs"),
            backstory=config.get("backstory", "You analyze requirements"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_retrieval_assistant(self):
        config = self.config.get("retrieval_assistant", {})
        return Agent(
            role=config.get("role", "Retrieval Specialist"),
            goal=config.get("goal", "Search resources"),
            backstory=config.get("backstory", "You find resources"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.library_tools.search_library]
        )
    
    def create_organization_assistant(self):
        config = self.config.get("organization_assistant", {})
        return Agent(
            role=config.get("role", "Organization Specialist"),
            goal=config.get("goal", "Evaluate resources"),
            backstory=config.get("backstory", "You organize information"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def create_collection_assistant(self):
        config = self.config.get("collection_assistant", {})
        return Agent(
            role=config.get("role", "Collection Assistant"),
            goal=config.get("goal", "Manage database"),
            backstory=config.get("backstory", "You manage resources"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.library_tools.check_book_exists, self.library_tools.add_to_library]
        )
    
    def create_recommendation_assistant(self):
        config = self.config.get("recommendation_assistant", {})
        return Agent(
            role=config.get("role", "Recommendation Specialist"),
            goal=config.get("goal", "Give recommendations"),
            backstory=config.get("backstory", "You recommend books"),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )