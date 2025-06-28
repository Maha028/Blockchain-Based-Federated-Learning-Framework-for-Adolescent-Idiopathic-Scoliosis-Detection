// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FederatedLearningLog
 * @dev Logs client updates and global model hashes for federated learning.
 */
contract FederatedLearningLog {

    // Struct to hold client update information
    struct ClientUpdate {
        string clientId;
        string modelHash;  // SHA256 hash of model weights
        uint roundId;
        uint trainAcc;
        uint valAcc;
        uint testAcc;
        uint precision;
        uint recall;
        uint f1;
        uint timestamp;
    }

    // Struct for global model update (optional)
    struct GlobalModelUpdate {
        string modelHash;
        uint roundId;
        uint timestamp;
    }

    // State variables
    ClientUpdate[] public clientUpdates;
    GlobalModelUpdate[] public globalModelUpdates;

    // Mapping for client-specific update counts (e.g., for tracking behavior)
    mapping(string => uint) public clientUpdateCounts;

    // Events
    event ClientUpdateLogged(
        string indexed clientId,
        string modelHash,
        uint roundId,
        uint timestamp
    );

    event GlobalModelUpdated(
        string modelHash,
        uint roundId,
        uint timestamp
    );
    // Reputation logic (simplified thresholds)
if (_valAcc >= 80 && _f1 >= 75) {
    reputation[_clientId] += 5;  // reward
} else if (_valAcc < 60 || _f1 < 50) {
    reputation[_clientId] -= 5;  // penalty
}
_clampReputation(_clientId);


    /**
     * @dev Logs a local client model update and its evaluation metrics.
     */
    function logClientUpdate(
        string memory _clientId,
        string memory _modelHash,
        uint _roundId,
        uint _trainAcc,
        uint _valAcc,
        uint _testAcc,
        uint _precision,
        uint _recall,
        uint _f1
    ) public {
        clientUpdates.push(ClientUpdate(
            _clientId,
            _modelHash,
            _roundId,
            _trainAcc,
            _valAcc,
            _testAcc,
            _precision,
            _recall,
            _f1,
            block.timestamp
        ));

        clientUpdateCounts[_clientId]++;
        emit ClientUpdateLogged(_clientId, _modelHash, _roundId, block.timestamp);
    }

    /**
     * @dev Logs a new global model hash after aggregation.
     */
    function logGlobalModelUpdate(
        string memory _modelHash,
        uint _roundId
    ) public {
        globalModelUpdates.push(GlobalModelUpdate(
            _modelHash,
            _roundId,
            block.timestamp
        ));

        emit GlobalModelUpdated(_modelHash, _roundId, block.timestamp);
    }

    /**
     * @dev Returns the number of client updates logged.
     */
    function getClientUpdateCount() public view returns (uint) {
        return clientUpdates.length;
    }

    /**
     * @dev Returns the number of global model updates logged.
     */
    function getGlobalUpdateCount() public view returns (uint) {
        return globalModelUpdates.length;
    }

    /**
     * @dev Returns a specific client update.
     */
    function getClientUpdate(uint index) public view returns (
        string memory, string memory, uint, uint, uint, uint, uint, uint, uint, uint
    ) {
        require(index < clientUpdates.length, "Invalid index");
        ClientUpdate memory u = clientUpdates[index];
        return (
            u.clientId, u.modelHash, u.roundId,
            u.trainAcc, u.valAcc, u.testAcc,
            u.precision, u.recall, u.f1, u.timestamp
        );
    }

    /**
     * @dev Returns a specific global model update.
     */
    function getGlobalModelUpdate(uint index) public view returns (
        string memory, uint, uint
    ) {
        require(index < globalModelUpdates.length, "Invalid index");
        GlobalModelUpdate memory g = globalModelUpdates[index];
        return (g.modelHash, g.roundId, g.timestamp);
    }
// Mapping to hold reputation scores
mapping(string => int) public reputation;

// Modifier to clamp reputation between 0 and 100
function _clampReputation(string memory clientId) internal {
    if (reputation[clientId] > 100) {
        reputation[clientId] = 100;
    } else if (reputation[clientId] < 0) {
        reputation[clientId] = 0;
    }
}

}
