const FederatedLearningLog = artifacts.require("FederatedLearningLog");

module.exports = function (deployer) {
  deployer.deploy(FederatedLearningLog);
};
