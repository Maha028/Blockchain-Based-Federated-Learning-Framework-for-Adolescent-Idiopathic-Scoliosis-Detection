pragma solidity ^0.5.0;
pragma experimental ABIEncoderV2; // Enable ABI Encoder V2 for handling complex data types like arrays


import "./Ownable.sol";

contract ModelUpdateLogger is Ownable {
    struct Model {
        string modelHash;
        uint256 accuracy;
        uint256 timestamp;
    }

    mapping(address => Model[]) public modelUpdates;

    event ModelUpdated(address indexed client, string modelHash, uint256 accuracy, uint256 timestamp);

    function logModelUpdate(string memory _modelHash, uint256 _accuracy) public {
        uint256 timestamp = now;
        modelUpdates[msg.sender].push(Model(_modelHash, _accuracy, timestamp));

        emit ModelUpdated(msg.sender, _modelHash, _accuracy, timestamp);
    }

    function getModelUpdates(address _client) public view returns (string[] memory, uint256[] memory, uint256[] memory) {
        uint256 updateCount = modelUpdates[_client].length;
        string[] memory hashes = new string[](updateCount);
        uint256[] memory accuracies = new uint256[](updateCount);
        uint256[] memory timestamps = new uint256[](updateCount);

        for (uint256 i = 0; i < updateCount; i++) {
            Model memory model = modelUpdates[_client][i];
            hashes[i] = model.modelHash;
            accuracies[i] = model.accuracy;
            timestamps[i] = model.timestamp;
        }

        return (hashes, accuracies, timestamps);
    }
}
