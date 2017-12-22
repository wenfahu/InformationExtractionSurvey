# InformationExtractionSurvey

## Learning Entity and Relation Embeddings for Knowledge Graph Completion

The key feature of knowledge graph completion is to predict link between to entities, that is, given two entities, a knowledge graph completion system would predict the relation between them. Knowledge graph completion is a key step in knowledge construction and it could be used to explore the hidden structural information of data. A knowledge graph completion system could be obtained via the supervision of existing knowledge graph and be used to find new relational facts of the entities. The task of knowledge completion is similar to that of social network analysis but more chanllenging as the node in the knowledge graph are entities with different types and attributes, also the edges of knowledge graph have specific types. Thus, the traditional methods for network (or graph) analysis are not quite promising for solving knowledge graph completion task. 

However,  embedding learning methods proves to achieve state of the art results for solving knowledge graph completion. This family of methods project a knowledge graph to a continuous vector space and preserve the specific information in the  mean time. The TransE model embed both entity and relation to the same vector space and assumes that the relationship between two entities corresponds to a translation between the embedding vectors of the entities, that is, for a fact triplet (h, r, t) where h, t are two entities and r the relation between them, h + r = t holds. Therefore, under TransE, the socre fucntion needs to be optimized is :

```
f_r(h,t) = || h + r -t ||^2_2
```

This model proposes a simple a effective way for solving knowledge graph prediction. However, this basic assumption also leads to the constraints of the relation prediction. TransE is only capable of modeling 1-to-1 relations. Therefore, TransH model is proposed to enable an entity having different representations when involved in various relations. TransH models the relation as a vector r on a hyperplane with w_r as the normal vector. For a face triplet (h,r,t), the entities h and t are first projected to the hyperplane of w_r as h_w, t_w, thus the score function is :

```
f_r(h,t) = || h_w + r_w -t ||^2_2
```

Both TransE and TransH embeds the entities and relations to the same vector space, while the TransR model proposed by this paper projects entities and relations to distinct vector spaces, that is entity space and multiple relation space( relation-specific entity spaces). For each face triplet (h,r,t), entities in the entity space are first projected into relation space as h_r and t_r. After the embedding, the translation that is similar to TransE and TransH is performed, that is, h_r + t = t_r.



# Neural Relation Extraction with Selective Attention over Instances
This paper proposes a sentence-level attention-based convolutional neural network for distant supervised relation extraction.
The overall structure of the sentence-level attentino-based CNN is shown in figure. The CNN is used as a feature extractor to embed the semantics of sentences. Thus, the relations can be represented as the composition of the sentence embeddings. Also, this work tries to solve the problem of wrong labelling in the training process of relation extraction system. Therefore, this paper proposes the sentence level attention over multiple instances to reduce the weights of noise instances. In the end, the obtained relation vector weighted by sentence level attention is much more robust than existing neural relation extraction models. The key parts of this model can be summaried as :
+ Sentence Encoder
+ Selective Attention over Instance

## Sentence Encoder
The input sentence is encoded into distributed vector representation by a CNN. Each word of the input sentence is transformed into a dense vector using word embeddings and further more each word in the sentence is assigned a position embedding based on the relative distance from current word to head or tail entities. As the word embedding has the semantic information of each single word and the position embedding addresses that the words close to target entities are usually informative to determine the relation between entities, the word embeddings and position embeddings are concatenated as the input of CNN.
## Selective Attention
The attention weights defined in the paper is as follows:
For a sentence embedding x_i
```
a_i = exp(e_i) / SUM(e_k)
```
where e_i is referred as a query-based function which scores how well the input sentence x_i and the predict relation r matches.
```
e_i = x_i A r
```
A is a weighted diagonal matrix, and r is the query vector associated with relation r. The attention weights and sentence embedding are multiplied to obtain the weighted sentence vector. And the weighted sentence vector is further fed into a softmax classifier for various relations.

# Adversarial Training for Relation Extraction
