# **`groupcast` Development Checklist**

**1. Models**:
- [ ] **Base Model (`BaseForecaster`)**:
    - [ ] Implement as a `pl.LightningModule`.
    - [ ] Define abstract methods for forward pass and training/validation/test steps.
- [ ] **Temporal Models**:
    - [ ] Support for variable sequence lengths.
    - [ ] LSTM-based forecaster.
    - [ ] GRU-based forecaster.
- [ ] **Single Agent Models**:
    - [ ] Implement models for individual agent trajectory predictions.
    - [ ] Example: MLP forecaster.
- [ ] **Multi-Agent Models**:
    - [ ] Implement architectures for multi-agent scenarios.
    - [ ] Support for Graph Neural Networks (GNNs).
- [ ] **Density Forecasters**:
    - [ ] Models for density predictions.
    - [ ] Support for Mixture Density Networks (MDNs).
- [ ] **Model Utilities**:
    - [ ] Saving/loading functionalities.
    - [ ] Implement transfer learning utilities.

---

**2. Datasets**:
- [ ] **Base Dataset**:
    - [ ] Define the core dataset template.
- [ ] **Dataset Loaders**:
    - [ ] Implement loaders for popular sports datasets.
    - [ ] Include a MockSingleAgentDataset for prototyping/testing.
- [ ] **Dataset Utilities**:
    - [ ] Data augmentation tools.
    - [ ] Normalization functionalities.
    - [ ] Handling missing data.

---

**3. Utilities**:
- [ ] **Evaluation**:
    - [ ] Methods to compute popular evaluation metrics for trajectory forecasting.
- [ ] **Visualization**:
    - [ ] Tools to visualize trajectories.
    - [ ] Heatmap generation.
- [ ] **Data Processing**:
    - [ ] Raw data preprocessing tools.

---

**4. Examples**:
- [ ] Provide Jupyter notebooks or Python scripts showcasing:
    - [ ] Basic package usage.
    - [ ] Advanced functionalities.
    - [ ] Tutorials on model training, evaluation, and inference.

---

**5. Documentation**:
- [ ] Create comprehensive docs:
    - [ ] Introduction to `groupcast`.
    - [ ] User guide.
    - [ ] Detailed API references for classes and methods.
    - [ ] Installation and setup guide.

---

**6. Tests**:
- [ ] Implement unit and integration tests:
    - [ ] Tests for model components.
    - [ ] Dataset loading and processing tests.
    - [ ] Utility functionality tests.

---

**Best Practices & Guidelines**:
- [ ] Revise this chatGPT generated checklist
- [ ] Use method prefixes in base classes.
- [ ] Ensure derived classes implement `abstractmethod` defined methods.
- [ ] Maintain modularity in key methods.
- [ ] Document all classes and methods.
- [ ] Implement comprehensive tests and maintain coverage.
- [ ] Version management and regular updates.