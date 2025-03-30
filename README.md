# CBML - Computational Biology Machine Learning

A machine learning framework for space biology research and biomedical datasets.

## Project Overview

This project aims to leverage machine learning techniques for space biology research by:
1. Curating biomedical datasets suitable for pre-training models
2. Identifying space biology-specific datasets from sources like NASA OSDR
3. Implementing transfer learning pipelines for space biology applications

## Current Implementation

The project currently includes:

- **Dataset Curation Module**: A framework for collecting and organizing biomedical datasets with metadata
- **Space Biology Dataset Module**: A system for identifying space biology-specific datasets
- **Transfer Learning Module**: Implementation of transfer learning techniques using PyTorch for space biology data

## Usage

```python
# Run the main pipeline
python test.py
```

This will:
1. Create and save a curated list of biomedical datasets
2. Create and save a list of space biology datasets
3. Run a demonstration of transfer learning using synthetic data
4. Save the trained model to `space_bio_model.pth`

## Dependencies

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- torchvision
- tqdm
- matplotlib

## Resources Used

- [PyTorch Documentation](https://pytorch.org/docs/)
- [NASA Open Science Data Repository](https://osdr.nasa.gov/)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Space Biology Research](https://www.nasa.gov/space-biology)

## TODO

- [ ] Acquire and integrate real biomedical datasets (noted as "DATA SET NAHI MIL RAHA" in code)
- [ ] Implement data preprocessing for space biology datasets
- [ ] Develop more specialized models for different space biology data types
- [ ] Create evaluation framework for model performance
- [ ] Add visualization tools for space biology data
- [ ] Implement advanced transfer learning techniques
- [ ] Create a proper dataset download module
- [ ] Add documentation and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- Siddhanth P Vashist
- Kinshuk Chauhan 
- Manobhav Sethi
- Sachin Pant