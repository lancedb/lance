from __future__ import annotations
import os

from label_studio_sdk import Client, Project
import pandas as pd


class LanceLabelStudioClient:
    """
    An API client for LabelStudio for importing/exporting Lance data
    """

    @classmethod
    def create(cls, url: str = None, api_key: str = None) \
            -> LanceLabelStudioClient:
        """
        Create a new LabelStudio API client

        Parameters
        ----------
        url: str, default None
            The LabelStudio server url.
            If None then look for "LABEL_STUDIO_SERVER_URL" envvar.
            If the envvar is empty, then default to http://localhost:8080
        api_key: str, default None
            If None then look for "LABEL_STUDIO_API_KEY" envvar

        Notes
        -----
        See labelstudio documentation https://labelstud.io/guide/sdk.html
        """
        if url is None:
            url = os.environ.get("LABEL_STUDIO_SERVER_URL",
                                 "http://localhost:8080")
        if api_key is None:
            api_key = os.environ["LABEL_STUDIO_API_KEY"]
        return LanceLabelStudioClient(url, api_key)

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.client = Client(url, api_key)
        self.client.check_connection()

    def get_project(self, name: str) -> LanceLabelStudioProject:
        """
        Get a LabelStudio Project by name
        """
        for p in self.client.get_projects():
            proj_name = p.get_params()['title'].lower().strip()
            if proj_name == name.lower().strip():
                return LanceLabelStudioProject(self.client, p)
        raise ValueError(f"Project {name} not found")


class LanceLabelStudioProject:
    """
    Wrapper with methods for importing/exporting Lance data
    from a Label Studio project
    """

    def __init__(self, client: Client, project: Project):
        """
        Parameters
        ----------
        client: Client
        project: Project
        """
        self.client = client
        self.project = project

    def add_tasks(self, df: pd.DataFrame,
                  image_col: str, pk_col: str,
                  label_col: str = None) -> list[str]:
        """
        Export Lance data to LabelStudio

        Parameters
        ----------
        df: pd.DataFrame
            Source DataFrame
        image_col: str
            Column name for the image uri's
        pk_col: str
            Column name for the data id
        label_col: str, default None
            Optional column for pre-annotation

        Returns
        -------
        ids: list[str]
            list of created task id's
        """
        tasks = []
        for i, row in df.iterrows():
            task_data = {'image': row[image_col], 'id': row[pk_col]}
            if label_col:
                task_data[label_col] = row[label_col]
            tasks.append(task_data)
        kwargs = {}
        if label_col:
            kwargs["preannotated_from_fields"] = [label_col]
        task_ids = self.project.import_tasks(tasks, **kwargs)
        return task_ids

    def get_annotations(self, label_col: str = "label",
                        pk_col: str = "id") -> pd.DataFrame:
        """
        Export data out of LabelStudio

        Parameters
        ----------
        label_col: str, default "label"
            The output label column name.
        pk_col: str, default "id"
            The name of the primary id key in the label studio data.
            Used as join key to merge into existing dataset
        """
        rs = []
        for t in self.project.get_tasks():
            label = t['annotations'][0]['result'][0]['value']['choices'][0]
            rs.append({pk_col: t['data'][pk_col], label_col: label})
        df = pd.DataFrame(rs)
        return df
