# Stream Scouter
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://placehold.co/800x200/EEE/31343C?font=poppins&text=Stream%20Scouter">
  <source media="(prefers-color-scheme: light)" srcset="https://placehold.co/800x200/31343C/EEE?font=poppins&text=Stream%20Scouter">
  <img alt="StreamScouter Logo" src="https://placehold.co/800x200/31343C/EEE?font=poppins&text=Stream%20Scouter">
</picture>

## Overview
Stream Scouter is a lightweight recommendation tool for matching Netflix titles to user queries using multilingual sentence transformers and semantic similarity. The project is designed to provide accurate and efficient content recommendations across different languages.

## Features
- 🚀 **Fast and Efficient**: Uses pre-trained sentence transformers for quick semantic matching.
- 🌍 **Multilingual Support**: Works with multiple languages for diverse audiences.
- 🔎 **Semantic Similarity Matching**: Finds the most relevant Netflix titles based on user input.
- 📊 **Notebook Implementation**: Fully implemented in Jupyter Notebook for ease of experimentation and deployment.

## Installation
To set up and run StreamScouter locally, follow these steps:

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required dependencies (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/r4nol/StreamScouter.git
   cd StreamScouter
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook and open `StreamScouter.ipynb`:
   ```bash
   jupyter notebook
   ```

## Usage
1. Run the cells in the Jupyter Notebook.
2. Enter a query (e.g., "After meeting at a party, a Cape Town teen tests a private-school swimmer's reputation") to get recommendations.
3. View the ranked list of matching Netflix titles.

## Technologies Used
- **Python** 🐍
- **Jupyter Notebook** 📓
- **Sentence Transformers** 🤖
- **Pandas & NumPy** 📊

## Contributing
We welcome contributions! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
🐙 GitHub: [@r4nol](https://github.com/r4nol)
