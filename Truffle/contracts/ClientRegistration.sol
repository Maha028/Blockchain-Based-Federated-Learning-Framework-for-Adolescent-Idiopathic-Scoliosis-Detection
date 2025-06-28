pragma solidity ^0.5.0;

import "./Ownable.sol";

contract ClientRegistration is Ownable {
    mapping(address => bool) public registeredClients;

    function registerClient(address _client) public onlyOwner {
        registeredClients[_client] = true;
    }

    function isClientRegistered(address _client) public view returns (bool) {
        return registeredClients[_client];
    }
}
