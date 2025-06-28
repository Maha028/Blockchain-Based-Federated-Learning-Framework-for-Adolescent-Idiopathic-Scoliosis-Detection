from web3 import Web3
import json
import os

# === CONFIGURATION ===
GANACHE_URL = "http://127.0.0.1:7545"

# Connect to Ganache
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
if not w3.isConnected():
    raise ConnectionError("⚠️ Web3 is not connected to Ganache")

# Load contract JSON (compiled by Truffle)
contract_path = os.path.join("blockchain", "build", "contracts", "FederatedLearningLog.json")
with open(contract_path) as f:
    contract_json = json.load(f)

abi = contract_json["abi"]
contract_address = contract_json["networks"]["5777"]["address"]  # Make sure network ID matches Ganache

# Create contract instance
contract = w3.eth.contract(address=contract_address, abi=abi)

# Set default account for transactions
w3.eth.default_account = w3.eth.accounts[0]

# === CONTRACT INTERACTIONS ===

def log_client_update(client_id, model_hash, round_id, train_acc, val_acc, test_acc, precision, recall, f1):
    """
    Logs a client's model update and metrics to the blockchain.
    """
    try:
        tx_hash = contract.functions.logClientUpdate(
            client_id, model_hash, round_id,
            int(train_acc), int(val_acc), int(test_acc),
            int(precision), int(recall), int(f1)
        ).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Client update logged on blockchain in tx: {receipt.transactionHash.hex()}")
        return receipt
    except Exception as e:
        print(f"❌ Error logging client update: {e}")
        return None

def log_global_model_update(model_hash, round_id):
    """
    Logs a global model update hash to the blockchain.
    """
    try:
        tx_hash = contract.functions.logGlobalModelUpdate(
            model_hash, round_id
        ).transact()
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Global model update logged in tx: {receipt.transactionHash.hex()}")
        return receipt
    except Exception as e:
        print(f"❌ Error logging global model update: {e}")
        return None

def get_client_reputation(client_id):
    """
    Retrieves current reputation score for a client.
    """
    try:
        score = contract.functions.reputation(client_id).call()
        print(f"📊 Reputation for {client_id}: {score}")
        return score
    except Exception as e:
        print(f"❌ Error fetching reputation: {e}")
        return -1

def get_total_client_updates():
    return contract.functions.getClientUpdateCount().call()

def get_client_update(index):
    return contract.functions.getClientUpdate(index).call()
