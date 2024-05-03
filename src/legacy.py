import os
import random
from legacy.legacy_analysis import run_FEVERISH_1, analyze_FEVERISH_1, run_FEVERISH_3, analyze_FEVERISH_3, run_FEVERISH_3_1, analyze_FEVERISH_3_1, run_feverish_3_2, analyze_feverish_3_2, run_feverish_3_3, analyze_feverish_3_3, run_feverish_3_4, analyze_feverish_3_4, run_feverish_3_5, analyze_feverish_3_5, run_feverish_3_6, analyze_feverish_3_6, run_feverish_3_7, analyze_feverish_3_7, run_final_version, analyse_final_version
from legacy.FEVERISH_1.db_loader import load as load_FEVERISH_1
from legacy.FEVERISH_1.FAISS_loader import load_FAISS
from legacy.FEVERISH_3.db_loader import main as load_FEVERISH_3
from legacy.FEVERISH_3_1.db_loader import main as load_FEVERISH_3_1
from legacy.FEVERISH_3_2.db_loader import main as load_FEVERISH_3_2
from legacy.FEVERISH_3_3.db_loader import main as load_FEVERISH_3_3

def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    claim_db_path = os.path.join(current_dir, '..', 'data')
    seed = "feverish"
    random.seed(seed)
    system_input = input("Enter the system name code\n1 - ESOTERIC 1\n3 - ESOTERIC 3\n3.1 - ESOTERIC 3.1\n 3.2 - ESOTERIC 3.2\n 3.3 - ESOTERIC 3.3\n 3.4 - ESOTERIC 3.4\n 3.5 - ESOTERIC 3.5\n 3.6 - ESOTERIC 3.6\n 3.7 - ESOTERIC 3.7\n final - Final version\n")
    if system_input == "1":
        FEVERISH_1(current_dir, claim_db_path, seed)
    elif system_input == "3":
        FEVERISH_3(current_dir, claim_db_path, seed)
    elif system_input == "3.1":
        FEVERISH_3_1(current_dir, claim_db_path, seed)
    elif system_input == "3.2":
        FEVERISH_3_2(current_dir, claim_db_path, seed)
    elif system_input == "3.3":
        FEVERISH_3_3(current_dir, claim_db_path, seed)
    elif system_input == "3.4":
        FEVERISH_3_4(current_dir, claim_db_path, seed)
    elif system_input == "3.5":
        FEVERISH_3_5(current_dir, claim_db_path, seed)
    elif system_input == "3.6":
        FEVERISH_3_6(current_dir, claim_db_path, seed)
    elif system_input == "3.7":
        FEVERISH_3_7(current_dir, claim_db_path, seed)
    elif system_input == "final":
        final_version(current_dir, claim_db_path, seed)
    else:
        print("Invalid system code")

def FEVERISH_1(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        file_ids = random.sample(range(1, 110), 5)
        print("Building database...")
        load_FEVERISH_1(file_ids)
        print("Loading FAISS...")
        # load_FAISS()
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_1', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_1')
        run_FEVERISH_1(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_1', 'output', 'FEVERISH_1.json')
        analyze_FEVERISH_1(file_path)
    else:
        print("Invalid action")

def FEVERISH_3(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        file_limit = 109
        print("Building database...")
        load_FEVERISH_3(file_limit)
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3')
        run_FEVERISH_3(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3', 'FEVERISH_3.json')
        analyze_FEVERISH_3(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_1(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        file_limit = 109
        print("Building database...")
        load_FEVERISH_3_1(file_limit)
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_1', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_1')
        run_FEVERISH_3_1(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_1', 'FEVERISH_3_1.json')
        analyze_FEVERISH_3_1(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_2(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        file_limit = 109
        print("Building database...")
        load_FEVERISH_3_2(file_limit)
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_2', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_2')
        run_feverish_3_2(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_2', 'FEVERISH_3_2.json')
        analyze_feverish_3_2(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_3(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        file_limit = 109
        print("Building database...")
        load_FEVERISH_3_3(file_limit)
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_3', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_3')
        run_feverish_3_3(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_3', 'FEVERISH_3_3.json')
        analyze_feverish_3_3(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_4(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        print("Can't build database for FEVERISH 3.4. Please use the provided database.")
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_4', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_4')
        run_feverish_3_4(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_4', 'FEVERISH_3_4.json')
        analyze_feverish_3_4(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_5(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        print("Can't build database for FEVERISH 3.5. Please use the provided database.")
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_5', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_5')
        run_feverish_3_5(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_5', 'FEVERISH_3_5.json')
        analyze_feverish_3_5(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_6(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        print("Can't build database for FEVERISH 3.6. Please use the provided database.")
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_6', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_6')
        run_feverish_3_6(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_6', 'FEVERISH_3_6.json')
        analyze_feverish_3_6(file_path)
    else:
        print("Invalid action")

def FEVERISH_3_7(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        print("Can't build database for FEVERISH 3.7. Please use the provided database.")
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_7', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_7')
        run_feverish_3_7(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_7', 'FEVERISH_3_7.json')
        analyze_feverish_3_7(file_path)
    else:
        print("Invalid action")

def final_version(current_dir, claim_db_path, seed):
    action = input("Enter the action you want to perform\n1 - Build \n2 - Run analysis\n3 - Show Stats\n")
    if action == "1":
        print("Can't build database for final version. Please use the provided database.")
    elif action == "2":
        print("Running analysis...")
        wiki_db_path = os.path.join(current_dir, 'legacy', 'final_version', 'data')
        output_dir = os.path.join(current_dir, 'legacy', 'final_version')
        run_final_version(claim_db_path, wiki_db_path, output_dir, seed)
    elif action == "3":
        print("Showing stats...")
        file_path = os.path.join(current_dir, 'legacy', 'final_version', 'final_version.json')
        analyse_final_version(file_path)
    else:
        print("Invalid action")

if __name__ == '__main__':
    main()
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # claim_db_path = os.path.join(current_dir, '..', 'data')
    # wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_3', 'data')
    # output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_3')
    # seed = "feverish"
    # run_feverish_3_3(claim_db_path, wiki_db_path, output_dir, seed)

    # wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_7', 'data')
    # output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_7')
    # run_feverish_3_7(claim_db_path, wiki_db_path, output_dir, seed)
    
    # wiki_db_path = os.path.join(current_dir, 'legacy', 'FEVERISH_3_6', 'data')
    # output_dir = os.path.join(current_dir, 'legacy', 'FEVERISH_3_6')
    # run_feverish_3_6(claim_db_path, wiki_db_path, output_dir, seed)