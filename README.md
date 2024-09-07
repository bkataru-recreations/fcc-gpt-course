# fcc gpt course

following [Create a Large Language Model from Scratch with Python – Tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8)

## datasets

### Wizard of Oz

download the Wizard of Oz book as plaintext `.txt` from [Dorothy and the Wizard in Oz by L. Frank Baum on Project Gutenberg](https://www.gutenberg.org/ebooks/22566)

```bash
$ curl -o wizard_of_oz.txt https://www.gutenberg.org/cache/epub/22566/pg22566.txt
```

### Open WebText

download the Open WebText corpus from huggingface

```bash
$ chmod +x openwebtext.sh
$ ./openwebtext.sh
```