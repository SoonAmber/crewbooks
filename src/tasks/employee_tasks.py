# src/tasks/employee_tasks.py
import yaml
from crewai import Task

class EmployeeTasks:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        with open("config/employee_tasks.yaml", "r") as file:
            return yaml.safe_load(file)

    def demand_translation(self, agent, query, reader_role="General Reader", reader_focus="Various Topics"):
        config = self.config["demand_translation"]
        return Task(
            description=config["description"].format(
                query=query, 
                reader_role=reader_role,
                reader_focus=reader_focus
            ),
            expected_output=config["expected_output"],
            agent=agent
        )

    def search(self, agent, requirements, demand_analysis, file_path="data/lite_library.csv", reader_role="", reader_focus=""):
        return Task(
            description=f"""
            Perform a comprehensive search for relevant resources based on demand "{requirements}":
            1. Search the local library catalog
            
            Based on the previous analysis ({demand_analysis}), gather all possible 
            materials related to the user's demand.
            Consider the reader's perspective: {reader_role} with focus on {reader_focus}
            
            First, analyze what key terms would be most relevant to search for based on the requirements.
            Then, search using those specific terms.
            If no results are found in the library, provide recommendations for books that could be added.
            """,
            expected_output="Output a list containing all findings, with each entry including resource details and source",
            agent=agent,
            context=[
                {
                    "role": "system",
                    "content": "When using tools, make sure to format the input as a properly formatted JSON string with only the query parameter."
                }
            ]
        )

    def employee_organization(self, agent, query, search_results, reader_role="General Reader", reader_focus="Various Topics"):
        config = self.config["employee_organization"]
        return Task(
            description=config["description"].format(
                query=query,
                reader_role=reader_role,
                reader_focus=reader_focus
            ),
            expected_output=config["expected_output"],
            agent=agent,
            context=[search_results]
        )

    def employee_collaboration(self, agent, query, organized_results, file_path="data/lite_library.csv"):
        config = self.config["employee_collaboration"]
        return Task(
            description=config["description"].format(
                query=query,
                file_path=file_path
            ),
            expected_output=config["expected_output"],
            agent=agent,
            context=[organized_results]
        )

    def results_and_feedback(self, agent, query, final_list, demand_analysis, reader_role="General Reader", reader_focus="Various Topics"):
        config = self.config["results_and_feedback"]
        return Task(
            description=config["description"].format(
                query=query,
                reader_role=reader_role,
                reader_focus=reader_focus,
                demand_analysis=demand_analysis
            ),
            expected_output=config["expected_output"],
            agent=agent,
            context=[final_list]
        )