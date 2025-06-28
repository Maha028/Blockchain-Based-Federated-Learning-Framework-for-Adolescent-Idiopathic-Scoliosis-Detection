const ModelUpdateLogger = artifacts.require("ModelUpdateLogger");

contract("ModelUpdateLogger", accounts => {
  let modelUpdateLogger;

  before(async () => {
    // Deploy the contract before running tests
    modelUpdateLogger = await ModelUpdateLogger.deployed();
  });

  it("should register a client", async () => {
    const clientAddress = accounts[1];
    
    // Register the client
    await modelUpdateLogger.registerClient(clientAddress, { from: accounts[0] });
    
    // Check if the client is registered
    const isRegistered = await modelUpdateLogger.registeredClients(clientAddress);
    assert.equal(isRegistered, true, "The client should be registered.");
  });

  it("should log a model update", async () => {
    const clientAddress = accounts[1];
    const modelHash = "0x123456789abcdef";
    const accuracy = 0.95;
    const trainingSamples = 1000;

    // Log a model update
    await modelUpdateLogger.updateModel(modelHash, accuracy, trainingSamples, { from: clientAddress });

    // Retrieve the model update
    const updates = await modelUpdateLogger.getModelUpdates(clientAddress);

    // Verify the logged model data
    assert.equal(updates[0].modelHash, modelHash, "Model hash should be logged correctly");
    assert.equal(updates[0].accuracy.toString(), accuracy.toString(), "Accuracy should be correct");
    assert.equal(updates[0].trainingSamples.toString(), trainingSamples.toString(), "Training samples should be correct");
  });
});
