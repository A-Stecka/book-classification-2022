# Classification of Book Genres Based on Their Summaries  
Book genre classification system created as part of the Artificial Intelligence and Knowledge Engineering course
-  
Data was obtained from the [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html).  

To classify a book into a specific genre based on its summary, two pieces of information are required: the plot summary of the book and a list of genres associated with the book. The summaries come from Wikipedia, so each book is assigned a Wikipedia article identifier from which its plot summary is derived. This identifier can serve as a key for a dictionary storing the data. The lists of genres associated with each book must also be stored. Additionally, although not crucial for the algorithm, the book title will also be stored to make the analyzed data understandable for a person interacting with the algorithm. All other data will not be analyzed in any way and can thus be disregarded.  

Consequently, the data is represented as a dictionary, in which:  
- the key is the Wikipedia article identifier,  
- the value is another dictionary:  
  - with the keys ‘title’, ‘genre’, ‘summary’,  
  - where the value is of type string.  
