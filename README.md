# Classification of Book Genres Based on Their Summaries  
Recognizing the genre of a book based on its summary was completed as part of the course Artificial Intelligence and Knowledge Engineering  
-  
Data was obtained from the [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html).  

To classify a book into a specific genre based on its summary, two pieces of information are essentially needed: the plot summary of the book and a list of genres to which the algorithm can classify the book. To obtain this list of genres, the input data must also read the genres for each book and process them accordingly. The summaries come from Wikipedia, so each book is assigned a Wikipedia article identifier from which its plot summary is derived. This identifier can serve as a key for a dictionary storing the data. Additionally, although not crucial for the algorithm, the book title will also be stored to make the analyzed data understandable for a person interacting with the algorithm. All other data will not be analyzed in any way and can thus be disregarded.  

Consequently, the data is represented as a dictionary, in which:  
- the key is the Wikipedia article identifier,  
- the value is another dictionary:  
  - with the keys ‘title’, ‘genre’, ‘summary’,  
  - where the value is of type string.  
