import pytest
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1] / 'src'))
from claim_generation.claim_generation import ClaimGenerator
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
from models import Claim, ClaimWrapper, Evidence, EvidenceWrapper

@pytest.fixture
def claim_generator():
    return ClaimGenerator()

@pytest.fixture
def evidence_retriever():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')
    return EvidenceRetriever(data_path)

@pytest.fixture
def claim():
    return Claim()