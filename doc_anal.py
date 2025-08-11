import logging
import requests
import time
from typing import Union, Dict
from config import key, endpoint, version


class DocumentIntelligenceService:
    """
    A service class for interacting with Azure Document Intelligence API.
    This class provides methods to analyze documents using Azure's Document Intelligence service.
    """

    def __init__(self):
        """
        Initialize the DocumentIntelligenceService with API credentials and endpoint.
        """
        self.key = key
        self.endpoint = endpoint
        self.api_version = version  # Currently only available in East US, West US2, and West Europe

    def analyze(
        self,
        source: Union[str, bytes],
        is_url: bool = True,
        model_id: str = "prebuilt-layout",
    ) -> Dict:
        """
        Analyze a document using Azure Document Intelligence.
        Args:
            source (Union[str, bytes]): The document source, either a URL or base64 encoded content.
            is_url (bool): True if the source is a URL, False if it's base64 encoded content.
            model_id (str): The ID of the model to use for analysis.
        Returns:
            Dict: The analysis results.
        Raises:
            requests.HTTPError: If the API request fails.
        """
        result_id = self._submit_analysis(source, is_url, model_id)
        return self._get_analysis_results(result_id, model_id)

    def _submit_analysis(
        self, source: Union[str, bytes], is_url: bool, model_id: str
    ) -> str:
        """
        Submit a document for analysis to Azure Document Intelligence.
        Args:
            source (Union[str, bytes]): The document source, either a URL or base64 encoded content.
            is_url (bool): True if the source is a URL, False if it's base64 encoded content.
            model_id (str): The ID of the model to use for analysis.
        Returns:
            str: The result ID for the submitted analysis.
        Raises:
            ValueError: If the Operation-Location header is missing in the response.
            requests.HTTPError: If the API request fails.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}:analyze?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.key,
        }
        data = {"urlSource": source} if is_url else {"base64Source": source}

        logging.info("Submitting document for analysis")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise ValueError("Operation-Location header is missing in the response.")

        return operation_location.split("/")[-1].split("?")[0]

    def _get_analysis_results(self, result_id: str, model_id: str) -> Dict:
        """
        Retrieve the analysis results from Azure Document Intelligence.
        Args:
            result_id (str): The ID of the analysis result to retrieve.
            model_id (str): The ID of the model used for analysis.
        Returns:
            Dict: The analysis results.
        Raises:
            requests.HTTPError: If the API request fails.
        """
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}/analyzeResults/{result_id}?api-version={self.api_version}&outputContentFormat=markdown"
        headers = {"Ocp-Apim-Subscription-Key": self.key}

        while True:
            logging.info("Waiting for analysis to complete.")
            time.sleep(2)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if data.get("status") in ["succeeded", "failed"]:
                return data

import json

if __name__ == "__main__":
    # Example usage of the DocumentIntelligenceService
    
    client = DocumentIntelligenceService()
    import time
    t1=time.time()
    analysis_results = client.analyze(
        source="https://hackrx.blob.core.windows.net/assets/Test%20/Test%20Case%20HackRx.pptx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A36%3A56Z&se=2026-08-05T18%3A36%3A00Z&sr=b&sp=r&sig=v3zSJ%2FKW4RhXaNNVTU9KQbX%2Bmo5dDEIzwaBzXCOicJM%3D"
    )
    t2=time.time()
    print("analysis_time",t2-t1)
    # Write the full JSON result to a file

    # Write just the list of top-level keys to a text file
    # with open("analysis_keys.txt", "w", encoding="utf-8") as f_keys:
    #     keys = analysis_results.keys()
    #     f_keys.write("Top-level keys in analysis_results:\n")
    #     for key in keys:
    #         f_keys.write(f"{key}\n")

    # Write keys inside analyzeResult
    analyze_result = analysis_results.get("analyzeResult", {})
    # with open("analyze_result_keys.txt", "w", encoding="utf-8") as f_ar_keys:
    #     f_ar_keys.write("Keys inside 'analyzeResult':\n")
    #     for key in analyze_result.keys():
    #         f_ar_keys.write(f"{key}\n")

    # Write the content text to a file
    content = analyze_result.get("content", "")
    with open("policy_ppt.txt", "w", encoding="utf-8") as f_content:
        f_content.write(content)

    # # Write a JSON dump of the tables extracted (if any)
    # tables = analyze_result.get("tables", [])
    # with open("tables.json", "w", encoding="utf-8") as f_tables:
    #     json.dump(tables, f_tables, indent=4, ensure_ascii=False)

    print("Analysis results have been saved to files:")
    print("- analysis_output.json")
    print("- analysis_keys.txt")
    print("- analyze_result_keys.txt")
    print("- content.txt")
    print("- tables.json")                 