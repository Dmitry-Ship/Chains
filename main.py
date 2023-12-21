from summary import pros_cons_generator
from coder import coder
from facts_extractor import facts_chain
from search import searcher
from suggester import suggester
from story import img2speach

article = """
Speculative evolution is a subgenre of science fiction and an artistic movement focused on hypothetical scenarios in the evolution of life, and a significant form of fictional biology.[1] It is also known as speculative biology[2] and it is referred to as speculative zoology[3] in regards to hypothetical animals.[1] Works incorporating speculative evolution may have entirely conceptual species that evolve on a planet other than Earth, or they may be an alternate history focused on an alternate evolution of terrestrial life. Speculative evolution is often considered hard science fiction because of its strong connection to and basis in science, particularly biology.
Speculative evolution is a long-standing trope within science fiction, often recognized as beginning as such with H. G. Wells's 1895 novel The Time Machine, which featured several imaginary future creatures. Although small-scale speculative faunas were a hallmark of science fiction throughout the 20th century, ideas were only rarely well-developed, with some exceptions such as Stanley Weinbaum's Planetary series, Edgar Rice Burroughs's Barsoom, a fictional rendition of Mars and its ecosystem published through novels from 1912 to 1941, and Gerolf Steiner's Rhinogradentia, a fictional order of mammals created in 1957.
The modern speculative evolution movement is generally agreed to have begun with the publication of Dougal Dixon's 1981 book After Man, which explored a fully realized future Earth with a complete ecosystem of over a hundred hypothetical animals. The success of After Man spawned several "sequels" by Dixon, focusing on different alternate and future scenarios. Dixon's work, like most similar works that came after them, were created with real biological principles in mind and were aimed at exploring real life processes, such as evolution and climate change, through the use of fictional examples.
"""

# article2 = """
# Broadly, the term liminal space is used to describe a place or state of change or transition; this may be physical (e.g. a doorway) or psychological (e.g. the period of adolescence).[3] Liminal space imagery often depicts this sense of "in-between", capturing transitional places (such as stairwells, roads, corridors, or hotels) unsettlingly devoid of people.[4] The aesthetic may convey moods of eeriness, surrealness, nostalgia, or sadness, and elicit responses of both comfort and unease.[5]
# Research by Alexander Diel and Michael Lewis of Cardiff University has attributed the unsettling nature of liminal spaces to the phenomenon of the uncanny valley. The term, which is usually applied to humanoids whose inexact resemblance to humans elicits feelings of unease, may explain similar responses to liminal imagery. In this case, physical places that appear familiar but subtly deviate from reality create the sense of eeriness typical of liminal spaces.[1]
# Peter Heft of Pulse: the Journal of Science and Culture further explores this sense of eeriness. Drawing on the works of Mark Fisher, Heft explains such eeriness may be felt when an individual views a situation in a different context to what they expect. For example, a schoolhouse, expected to be a busy amalgamation of teachers and students, becomes unsettling when depicted as unnaturally empty. This "failure of presence" was considered by Fisher to be one of the hallmarks of the aesthetic experience of eeriness.[2]
# """
# facts_chain.invoke({"text_input": article})
# pros_cons_generator.invoke({"input": "white board interview"})
# coder.invoke({"input": "is bird a palindrome?"})
# searcher.invoke({"input": "I'd like to figure out what games are tonight in NYC"})
# suggester.invoke('What are your favorite movie genres?')

img2speach('https://images.photowall.com/products/69237/lion-close-up.jpg?h=699&q=85')