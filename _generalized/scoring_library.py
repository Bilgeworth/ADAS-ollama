import json
from typing import Any, Callable, Literal
import openai
import backoff

class ScoringLibrary:
    def __init__(self, llm_base_url: str = 'http://localhost:11434/v1', llm_api_key: str = 'ollama'):
        self.client = openai.OpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
        )

    def get_match_score(
        self,
        expected: Any,
        received: Any,
        comparison_type: Literal["matrix", "string", "llm_judge"],
        **kwargs
    ) -> float:
        """
        Calculate a match score between expected and received data using the specified comparison type.
        
        cmd_line_args:
        expected: The expected (correct) data
        received: The received (to be evaluated) data
        comparison_type: The type of comparison to perform
        **kwargs: Additional arguments for specific comparison functions
        
        Returns:
        float: The calculated match score
        """
        comparison_functions = {
            "matrix": self.matrix_comparison,
            "string": self.string_comparison,
            "llm_judge": self.llm_judge_comparison
        }
        
        if comparison_type not in comparison_functions:
            raise ValueError(f"Unsupported comparison type: {comparison_type}")
        
        comparison_function = comparison_functions[comparison_type]
        
        try:
            return comparison_function(expected, received, **kwargs)
        except Exception as e:
            print(f"Error in comparison: {e}")
            return 0.0

    @staticmethod
    def matrix_comparison(expected: list[list[Any]], received: list[list[Any]]) -> float:
        """
        Compare two matrices and return a score based on matching elements.
        """
        if not received:
            return 0.0
        
        score = 0
        total_elements = 0
        
        for i, row in enumerate(expected):
            for j, element in enumerate(row):
                total_elements += 1
                try:
                    if received[i][j] == element:
                        score += 1
                except IndexError:
                    pass
        
        return score / total_elements if total_elements > 0 else 0.0

    @staticmethod
    def string_comparison(expected: str, received: str) -> float:
        """
        Compare two strings and return a score based on matching characters.
        """
        return len(set(expected) & set(received)) / len(set(expected))

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def api_call_initial(
        self,
        msg: str,
        model: str,
        system_message: str,
        temperature: float = 0.5
    ) -> dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ],
            temperature=temperature, 
            max_tokens=1024, 
            stop=None, 
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        json_dict = json.loads(content)
        assert json_dict is not None
        return json_dict

    def llm_judge_comparison(
        self, 
        expected: str, 
        received: str, 
        model: str = "llama2", 
        criteria: str = "relevance and accuracy"
    ) -> float:
        """
        Use an LLM to judge the similarity between expected and received text.
        """
        system_message = f"""
        You are an AI judge tasked with evaluating the similarity between two pieces of text.
        You will be given the expected (correct) text and the received (to be evaluated) text.
        Your job is to rate the similarity on a scale from 0 to 1, where 1 means perfect match
        and 0 means completely different. Focus on {criteria}.
        Provide your response as a JSON object with a single key 'score' and the float value.
        """
        
        msg = f"""
        Expected text: {expected}
        Received text: {received}
        
        Please evaluate the similarity and provide a score between 0 and 1.
        """
        
        result = self.api_call_initial(msg, model, system_message)
        return float(result['score'])

# Example usage:
if __name__ == "__main__":
    scorer = ScoringLibrary()
    
    # Matrix comparison
    expected_matrix = [[1, 2, 3], [4, 5, 6]]
    received_matrix = [[1, 2, 3], [4, 5, 0]]
    matrix_score = scorer.get_match_score(expected_matrix, received_matrix, "matrix")
    print(f"Matrix match score: {matrix_score}")
    
    # String comparison
    expected_string = "hello"
    received_string = "hallo"
    string_score = scorer.get_match_score(expected_string, received_string, "string")
    print(f"String match score: {string_score}")
    
    # LLM judge comparison
    expected_text = "The quick brown fox jumps over the lazy dog."
    received_text = "A fast brown fox leaps above a sleepy canine."
    llm_score = scorer.get_match_score(expected_text, received_text, "llm_judge", model="llama2", criteria="semantic similarity")
    print(f"LLM judge score: {llm_score}")