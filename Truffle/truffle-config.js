module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",    // Localhost (default: none)
      port: 8484,           // Ganache default port
      network_id: "*",      // Match any network id
    },
  },
  compilers: {
    solc: {
      version: "0.8.x",     // Use the specific 0.5.x version compatible with your contracts
      settings: {
        optimizer: {
          enabled: true,     // Enable optimizer (optional)
          runs: 200,         // Set optimization runs (optional)
        },
      },
    },
  },
};
