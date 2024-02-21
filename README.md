# nlp-automated-fact-checker
A Flask-based website that enables users to upload their Apple Music playlists and generate playlist names using GPT-3.5. The Spotify API is used to connect to the user's account and retrieve both their playlists and the songs inside each playlist. The OpenAI API text-davinci-003 model is used to generate names for the playlists based on the songs inside them. 

## Installation
Clone repository

```bash
git clone https://github.com/theobaur13/nlp-automated-fact-checker
```
Set up virtual environment

```python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```python
venv\Scripts\activate
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
