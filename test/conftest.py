import pytest
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / 'src'))
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
from models import Evidence, EvidenceWrapper

@pytest.fixture
def evidence_retriever():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')
    return EvidenceRetriever(data_path)