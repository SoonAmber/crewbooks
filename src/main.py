from dotenv import load_dotenv
import os
import time
from crewai import Crew, Agent, Task, Process, LLM

from src.agents.reader_agents import ReaderAgents
from src.agents.employee_agents import EmployeeAgents
from src.tasks.reader_tasks import ReaderTasks
from src.tasks.employee_tasks import EmployeeTasks
from src.utils.library_tools import LibraryTools

load_dotenv()

class LibrarySystem:
    def __init__(self, model_name="ollama/llama3"):
        # Use Ollama for LLM if model name starts with "ollama/"
        if model_name.startswith("ollama/"):
            self.llm = LLM(model=model_name, base_url="http://localhost:11434")
        else:
            # Otherwise use OpenAI or other models
            self.llm = LLM(model=model_name)
        
        # Initialize tools
        self.library_tools = LibraryTools()
        
        # Initialize agents
        self.reader_agents = ReaderAgents(self.llm)
        self.employee_agents = EmployeeAgents(self.llm, self.library_tools)
        
        # Initialize tasks
        self.reader_tasks = ReaderTasks()
        self.employee_tasks = EmployeeTasks()
    
    def safe_get_output(self, crew_output, task_id=None, index=0):
        """Safely extract output from crew results, handling different crewAI versions"""
        try:
            if crew_output is None:
                return "No output received from task"
                
            # If we got a string directly, just return it
            if isinstance(crew_output, str):
                return crew_output
                
            # Handle dictionary format (newer crewAI)
            elif isinstance(crew_output, dict):
                # Check if it's a CrewOutput object with a 'results' attribute
                if hasattr(crew_output, 'results') and isinstance(crew_output.results, dict):
                    if task_id and task_id in crew_output.results:
                        return crew_output.results[task_id]
                    elif crew_output.results:
                        return next(iter(crew_output.results.values()))
                        
                # Regular dictionary
                if task_id and task_id in crew_output:
                    return crew_output[task_id]
                elif crew_output:
                    return next(iter(crew_output.values()))
                
                return "No output found in task results"
            
            # Handle list format (older crewAI)
            elif isinstance(crew_output, list):
                if index < len(crew_output):
                    return crew_output[index]
                return "Index out of range in task results"
                
            # If it's an object with a string representation, convert it
            else:
                return str(crew_output)
                
        except Exception as e:
            print(f"Error extracting task output: {e}")
            return f"Error processing task output: {str(e)}"
    
    def run_reader_crew(self, topic):
        try:
            print("🧠 Creating reader agents...")
            # Create agents
            knowledge_expander = self.reader_agents.create_knowledge_expander()
            inherent_knowledge_keeper = self.reader_agents.create_inherent_knowledge_keeper()
            multidimensional_integrator = self.reader_agents.create_multidimensional_integrator()
            
            print("📝 Creating reader tasks...")
            # Create question tasks
            expander_question_task = self.reader_tasks.reader_question(knowledge_expander, topic)
            keeper_question_task = self.reader_tasks.reader_question(inherent_knowledge_keeper, topic)
            integrator_question_task = self.reader_tasks.reader_question(multidimensional_integrator, topic)
            
            # Run initial question tasks
            print("🚀 Running initial question tasks...")
            initial_crew = Crew(
                agents=[knowledge_expander, inherent_knowledge_keeper, multidimensional_integrator],
                tasks=[expander_question_task, keeper_question_task, integrator_question_task],
                verbose=True,
                process=Process.sequential
            )
            
            questions_output = initial_crew.kickoff()
            print(f"Questions output type: {type(questions_output)}")
            
            expander_question = self.safe_get_output(questions_output, expander_question_task.id, 0)
            keeper_question = self.safe_get_output(questions_output, keeper_question_task.id, 1)
            integrator_question = self.safe_get_output(questions_output, integrator_question_task.id, 2)
            
            # Create book description tasks
            expander_description_task = self.reader_tasks.book_description(
                knowledge_expander, expander_question
            )
            keeper_description_task = self.reader_tasks.book_description(
                inherent_knowledge_keeper, keeper_question
            )
            integrator_description_task = self.reader_tasks.book_description(
                multidimensional_integrator, integrator_question
            )
            
            # Run book description tasks
            print("🚀 Running book description tasks...")
            description_crew = Crew(
                agents=[knowledge_expander, inherent_knowledge_keeper, multidimensional_integrator],
                tasks=[expander_description_task, keeper_description_task, integrator_description_task],
                verbose=True,
                process=Process.sequential
            )
            
            descriptions_output = description_crew.kickoff()
            print(f"Descriptions output type: {type(descriptions_output)}")
            
            expander_description = self.safe_get_output(descriptions_output, expander_description_task.id, 0)
            keeper_description = self.safe_get_output(descriptions_output, keeper_description_task.id, 1)
            integrator_description = self.safe_get_output(descriptions_output, integrator_description_task.id, 2)
            
            # Format final requirements
            print("📋 Formatting final requirements...")
            format_task = self.reader_tasks.format_final_requirements(
                multidimensional_integrator,  # Using integrator to synthesize perspectives
                expander_description,      # Expander's perspective
                keeper_description,      # Keeper's perspective
                integrator_description       # Integrator's perspective
            )
            
            format_crew = Crew(
                agents=[multidimensional_integrator],
                tasks=[format_task],
                verbose=True
            )
            
            format_output = format_crew.kickoff()
            formatted_requirements = self.safe_get_output(format_output, format_task.id, 0)
            
            return formatted_requirements
        except Exception as e:
            print(f"Error in reader crew: {e}")
            # Fallback formatted requirements
            return f"""
            1. Requirement for exploring emerging trends in {topic}.
            2. Requirement for understanding core principles of {topic}.
            3. Requirement for connecting {topic} with other disciplines.
            """
    
    def run_employee_crew(self, requirements, topic):
        try:
            print("👥 Creating employee agents...")
            # Create agents
            demand_assistant = self.employee_agents.create_demand_assistant()
            retrieval_assistant = self.employee_agents.create_retrieval_assistant()
            organization_assistant = self.employee_agents.create_organization_assistant()
            collection_assistant = self.employee_agents.create_collection_assistant()
            recommendation_assistant = self.employee_agents.create_recommendation_assistant()
            
            print("📋 Analyzing requirements...")
            # Analyze requirements
            demand_task = self.employee_tasks.demand_translation(
                demand_assistant, 
                requirements,
                reader_role="Topic Explorer", 
                reader_focus=topic
            )
            
            demand_crew = Crew(
                agents=[demand_assistant],
                tasks=[demand_task],
                verbose=True
            )
            
            demand_output = demand_crew.kickoff()
            demand_analysis = self.safe_get_output(demand_output, demand_task.id, 0)
            
            print("🔍 Searching for resources...")
            # Search for resources
            search_task = self.employee_tasks.search(
                retrieval_assistant,
                requirements,
                demand_analysis,
                file_path="data/lite_library.csv",
                reader_role="Topic Explorer",
                reader_focus=topic
            )
            
            search_crew = Crew(
                agents=[retrieval_assistant],
                tasks=[search_task],
                verbose=True
            )
            
            search_output = search_crew.kickoff()
            print(f"Search output type: {type(search_output)}")
            # Add more defensive handling here
            search_results = self.safe_get_output(search_output, search_task.id, 0)
            
            print("📊 Organizing search results...")
            # Organize results
            organization_task = self.employee_tasks.employee_organization(
                organization_assistant,
                requirements,
                search_results,
                reader_role="Topic Explorer",
                reader_focus=topic
            )
            
            organization_crew = Crew(
                agents=[organization_assistant],
                tasks=[organization_task],
                verbose=True
            )
            
            organization_output = organization_crew.kickoff()
            organized_results = self.safe_get_output(organization_output, organization_task.id, 0)
            
            print("📚 Managing library collection...")
            # Collection management
            collection_task = self.employee_tasks.employee_collaboration(
                collection_assistant,
                requirements,
                organized_results,
                file_path="data/lite_library.csv"
            )
            
            collection_crew = Crew(
                agents=[collection_assistant],
                tasks=[collection_task],
                verbose=True
            )
            
            collection_output = collection_crew.kickoff()
            collection_results = self.safe_get_output(collection_output, collection_task.id, 0)
            
            print("📝 Generating final recommendations...")
            # Final recommendations
            recommendation_task = self.employee_tasks.results_and_feedback(
                recommendation_assistant,
                requirements,
                collection_results,
                demand_analysis,
                reader_role="Topic Explorer",
                reader_focus=topic
            )
            
            recommendation_crew = Crew(
                agents=[recommendation_assistant],
                tasks=[recommendation_task],
                verbose=True
            )
            
            recommendation_output = recommendation_crew.kickoff()
            final_recommendations = self.safe_get_output(recommendation_output, recommendation_task.id, 0)
            
            return final_recommendations
        except Exception as e:
            print(f"Error in employee crew: {e}")
            return f"Based on your interest in {topic}, we recommend exploring our catalog for related books."
    
    def recommend_books(self, topic):
        try:
            print(f"🎯 Processing topic: {topic}")
            
            # Run reader crew
            print("\n==== READER ANALYSIS PHASE ====")
            requirements = self.run_reader_crew(topic)
            print(f"\n📋 Requirements generated:\n{requirements}\n")
            
            # Run employee crew
            print("\n==== EMPLOYEE RECOMMENDATION PHASE ====")
            recommendations = self.run_employee_crew(requirements, topic)
            print(f"\n📚 Book recommendations ready!")
            
            return {
                "requirements": requirements,
                "recommendations": recommendations
            }
        except Exception as e:
            print(f"❌ Error in recommendation process: {e}")
            return {
                "requirements": f"Could not generate requirements for {topic}.",
                "recommendations": f"We encountered an error while generating recommendations for {topic}. Please try again or choose a different topic."
            }