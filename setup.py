
from cltk import NLP
import re
   
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx
import re




from lxml import etree

import networkx as nx
import matplotlib.pyplot as plt

from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
cltk_greek = NLP(language='grc', suppress_banner=True)

#create dictionary for all character-dictionaries
dict_of_dicts = {}



def divide_sections(text):
    tree = etree.parse(text)
    total_characters = []
    root = tree.getroot()
    for elem in root.findall('content'):
        """  book = elem.find('book').text
        chapter = elem.find('chapter').text """
        section = elem.find('text').text
        characters = find_characters(section)
        total_characters.append(characters)
    total_characters = ' '.join(total_characters)
    total_characters = word_tokenize(total_characters)
    total_set = set(total_characters)
    return total_set


def find_characters(section):
    characters_list = []
    cltk_doc = cltk_greek.analyze(section)
    for sent in cltk_doc.sentences:
            for word in sent:
                if word.upos == 'PROPN':
                    characters_list.append(word.string)
    return ' '.join(characters_list)



  
#get full list of characters      
result_longus = divide_sections('/Users/emeliehallenberg/cltk_data/Daphnis_et_Chloe.xml')
#print(result_longus)


#make a list of result
full_characters_longus = ['Δευτέρας', 'Νάπη', 'Εὑροῦσα', 'Ἰδοῦσα', 'Ἴτυν', 'Ἀγχίσης', 'Στρατιώτου', 'Εὐδρόμου', 
                          'Διονύσου', 'Πᾶνα', '—ἀναπηδᾷ', 'Μέλλοντος', 'Τέως', 'Διονυσοφάνης', 'Ἄστυλε', 'Δραμὼν', 
                          'Ἵππασον', 'Διονυσιακὸν', 'Ἀριάδνην', 'Δάφνις·', 'Ἄνεμος', 'Λύκαινα', 'Δόρκωνα', 'Χλόη·',
                          'Πενθέα', 'Νάπην', 'Τέτμητο', 'Λυκοῦργον', 'Νάπης', 'Πίτυος', 'Μυρτάλην', 'Ἐπιμηλίσι', 
                          'Κλεαρίστῃ', 'Πίτυν', 'Πανός—', 'Νάπῃ', 'Πὰν', 'Κρέως', 'Ἀπόλλων', 'Ἀμφοτέρους', 'Δόρκωνι', 
                          'Δρόμῳ', 'Οἴνου', 'Γναθωνάριον', 'Δρύαντι', 'Σκέψις', 'Διὸς', 'Δάφνιδος·', 'Δρύαντα·', 
                          'Ἄστυλος', 'Τίτυρος', 'Τέτρωτο', 'Δάκρυα', 'Γνάθωνος', 'Ἀμέλει', 'Διόνυσον', 'Μαθέτω', 
                          'Ἀθύρματα', 'Λάμωνα', 'Βάκχην', 'Μελίαι', 'Ἀστύλου', 'Βράγχος', 'Δρυμὸς', 'Διονύσῳ', 
                          'Θαύματι', 'Ἔρωτος·', 'Διονυσοφάνην', 'Ἀμαρυλλίδα', 'Γνάθων', 'Ῥόδῃ', 'Καμὼν', 'Δάφνις',
                          'Ἔρωτος', 'Σύμβολον', 'Ἀμαρυλλίδος', 'Δάφνι', 'Τυχὼν', 'Λάμπις', 'Δάφνιδι', 'Σιωπηλὸς', 
                          'Μυρτάλῃ', 'Μέμνησό', 'Κρεμᾷ', 'Χαιρέτω·', 'Κρόνου', 'Ποιμένος', 'Χλόην', 'Ἀμαρυλλίδος·',
                          'Δρυάσιν', 'Πανί', 'Ἠχοῦς', 'Μαρσύαν·', 'Χρῶμις', 'Διόνυσε', 'Τύχης', 'Χλόην·', 'Αἶγα', 
                          'Διονυσοφάνει', 'Οἰκτείρει', 'Διὶ', 'Δρύαντα', 'Σχοῖνον', 'Θᾶττον', 'Γνάθωνα', 'Λέσβῳ', 
                          'Δάφνιν·', 'Μούσαις', 'Εὔδρομος', 'Λαομέδοντι', 'Ἵππασος', 'Ζεύς', 'Λάμων', 'Μιτυλήνῃ', 
                          'Προσάγει', 'Πανὸς', 'Ὀνειροπολεῖ', 'Βουκόλος', 'Ἑρμῇ', 'Βιαίου', 'Δυσθήρατος', 'Τερπνὸν',
                          'Σκιά', 'Γανυμήδης', 'Εὔδρομον', 'Μιτυλήνη', 'Πανὶ', 'Ἀγέλην', 'Μυρτάλη·', 'Κἀκείνη', 
                          'Μυρτάλη', 'Κλεαρίστης', 'Διόνυσος', 'Μεγακλέους', 'Χλόης', 'Χλόη', 'Δάφνι·', 'Φιλητᾶν',
                          'Λέσβου', 'Λυκαίνιον·', 'Φθανούσης', 'Λύκος', 'Δία·', 'Δρύα', 'Μεγακλῆς', 'Μετοπώρου', 
                          'Κλεαρίστη', 'Μυρτάλης', 'Λέσβον', 'Δάφνιδος', 'Συνετὰ', 'Δόρκωνος', 'Ἔνδον', 'Πηγὰς', 
                          'Λυκαίνιον', 'Ἀγένειός', 'Χλόῃ', 'Σωτῆρι', 'Λύκου', 'Χρόνος', 'Σεμέλην', 'Μέμνησο', 
                          'Λάμωνι', 'Μηθύμνης', 'Νύμφας', 'Φιλητᾶς', 'Δέομαι', 'Χλόης·', 'Μιτυλήνην', 'Δρύαντος',
                          'Ἀστύλῳ', 'Κάμνε', 'Τίτυρον', 'Μεγακλεῖ', 'Λάμπιδος', 'Νυμφῶν', ';', 'Δάφνιν', 'Σῦριγξ',
                          'Κοῖλος', 'Σύριγγος·', 'Ῥόδην', 'Χαίρων', 'Λάμωνος', 'Κρύος', 'Διόνυσος·', 'Μιτυλήνης', 
                          'Ἀνάστα', 'Ἀφροδίτη·', 'Ἀφροδίτη', 'Ἄστυλον', 'Ἠμέλητο', 'Φιλητᾶ', 'Φιλοποίμενα', 
                          'Νύμφαις', 'Νύμφαι', 'Κῆπός', 'Φυλάττει', 
                          'Θάρρει', 'Δρύας', 'Εὐλίμενός', 'Δήμητρι', 'Ταχείας', 'Σύριγγα', 'Λάμπιν', 
                          'Θηράσων', 
                          'Σικελὸς', 'Χρῶμιν']


#remove characters that are mentioned less than 5 times
def count_chars(text, li):
    tree = etree.parse(text)
    fulltext = ''
    root = tree.getroot()
    for elem in root.findall('content'):
        section = elem.find('text').text
        fulltext += section
    for char in li:
        sum = fulltext.count(char)
        if sum > 5:
            print(char, sum)


#count_chars('/Users/emeliehallenberg/cltk_data/Daphnis_et_Chloe.xml', full_characters_longus)

#manually create list of final characters with all name forms
final_list_chars = ['Νάπη', 'Νάπῃ', 'Νάπης', 'Νάπην',
                    'Διονυσοφάνης', 'Διονυσοφάνην', 'Διονυσοφάνει',
                    'Πὰν', 'Πανί', 'Πανὸς', 'Πᾶνα',
                    'Ἄστυλος', 'Ἄστυλον', 'Ἄστυλε',
                    'Γνάθων', 'Γνάθωνος', 'Γνάθωνα',
                    'Δάφνις', 'Δάφνιδος', 'Δάφνιδι', 'Δάφνι', 'Δάφνιν',
                    'Λάμων', 'Λάμωνος', 'Λάμωνι', 'Λάμωνα',
                    'Μυρτάλη', 'Μυρτάλης',  'Μυρτάλῃ', 'Μυρτάλην',
                    'Χλόη', 'Χλόης', 'Χλόην', 'Χλόῃ',
                    'Φιλητᾶς', 'Φιλητᾶν', 'Φιλητᾶ',
                    'Νύμφαι', 'Νύμφαις', 'Νύμφας',
                    'Δρύας', 'Δρύαντος', 'Δρύα', 'Δρύαντα',
                    'Λυκαίνιον', 
                    'Δόρκωνος', 'Δόρκωνι', 'Δόρκωνα', 'Δόρκων'
                    ]

#add a dictionary for all name forms
for c in final_list_chars:
    dict_of_dicts[c] = {}


#iterate through sections again
def clean_sections(text):
    clean_sections = []
    tree = etree.parse(text)
    root = tree.getroot()
    for elem in root.findall('content'):
        section = elem.find('text').text
        clean_sections.append(section)
    return clean_sections


#sections_longus = clean_sections('/Users/emeliehallenberg/cltk_data/Daphnis_et_Chloe.xml')


#find co-occurrences in every section
def find_coocs(sections, char, li):
    char_dict = {}
    for char_2 in li:
        for sect in sections:
            if char_2 in sect and char in sect:
                if char_2 in char_dict:
                    char_dict[char_2] += 1
                else:
                    char_dict[char_2] = 1
                sect.replace(char, '')
    return char_dict
        


#iterate through every name form in the list, and add resulting dictionaries to the dict of dictionaries
""" for char in final_list_chars:
    dict_result = find_coocs(sections_longus, char, final_list_chars)
    dict_of_dicts[char] = dict_result

for k, v in dict_of_dicts.items():
    print(k, v)  """ 

#manually remove duplicates and add resulting co-occurrences together 
dict_of_dicts = {'Νάπη': {'Δάφνις': 15, 'Λάμων': 11, 'Μυρτάλη': 7, 'Χλόη': 16, 'Φιλητᾶς': 2, 'Νύμφαι': 4, 'Δρύας': 25},
'Διονυσοφάνης': {'Γνάθων': 3, 'Δάφνις': 14, 'Λάμων': 2, 'Χλόη': 6, 'Δρύας': 5, 'Ἄστυλος': 1, 'Νύμφαι': 1},
'Πὰν': {'Δάφνις': 18, 'Λάμων': 4, 'Χλόη': 19, 'Φιλητᾶς': 7, 'Νύμφαι': 17, 'Γνάθων': 1},
'Ἄστυλος': {'Διονυσοφάνης': 1, 'Γνάθων': 6, 'Δάφνις': 17, 'Λάμων': 7, 'Μυρτάλη': 3},
'Γνάθων': {'Διονυσοφάνης': 3, 'Πὰν': 1, 'Ἄστυλος': 6, 'Δάφνις': 25, 'Λάμων': 11, 'Μυρτάλη': 1, 'Νύμφαι': 1, 'Δρύας': 2},
'Δάφνις': {'Νάπη': 15,  'Γνάθων': 25, 'Διονυσοφάνης': 14, 'Πὰν': 18, 'Ἄστυλος': 17, 'Λάμων': 49, 'Μυρτάλη': 28, 'Χλόη': 297, 'Φιλητᾶς': 26, 'Νύμφαι': 31, 'Δρύας': 71},
'Λάμων': {'Νάπη': 11, 'Ἄστυλος': 3, 'Διονυσοφάνης': 2, 'Πὰν': 4, 'Ἄστυλος': 4, 'Γνάθων': 11, 'Δάφνις': 49, 'Μυρτάλη': 33, 'Χλόη': 25, 'Φιλητᾶς': 6, 'Νύμφαι': 15, 'Δρύας': 32},
'Μυρτάλη': {'Νάπη': 7, 'Ἄστυλος': 3, 'Δάφνις': 30, 'Γνάθων': 1, 'Λάμων': 33, 'Χλόη': 4, 'Φιλητᾶς': 2, 'Νύμφαι': 1, 'Δρύας': 12},
'Χλόη': {'Νάπη': 16, 'Διονυσοφάνης': 6, 'Πὰν': 14, 'Δάφνις': 297, 'Λάμων': 25, 'Μυρτάλη': 4, 'Φιλητᾶς': 6, 'Νύμφαι': 37, 'Δρύας': 41},
'Φιλητᾶς': {'Νάπη': 2, 'Πὰν': 7, 'Δάφνις': 26, 'Λάμων': 6, 'Μυρτάλη': 2, 'Χλόη': 6, 'Νύμφαι': 6, 'Δρύας': 8},
'Νύμφαι': {'Νάπη': 4, 'Πὰν': 17, 'Διονυσοφάνης': 1, 'Δάφνις': 31, 'Γνάθων': 1, 'Λάμων': 15, 'Μυρτάλη': 1, 'Χλόη': 37, 'Φιλητᾶς': 7, 'Δρύας': 12},
'Δρύας': {'Νάπη': 25, 'Διονυσοφάνης': 5, 'Δάφνις': 72, 'Γνάθων': 2, 'Λάμων': 27, 'Μυρτάλη': 12, 'Χλόη': 46, 'Φιλητᾶς': 8, 'Νύμφαι': 12},
'Λυκαίνιον': {'Νάπη': 1, 'Δάφνις': 12, 'Λάμων': 1, 'Μυρτάλη': 1, 'Χλόη': 6, 'Φιλητᾶς': 2, 'Δρύας': 2,  'Δόρκων': 3},
'Δόρκων': {'Δάφνις': 29, 'Χλόη': 21, 'Νύμφαι': 2, 'Νάπη': 2, 'Λάμων': 2, 'Μυρτάλη': 2, 'Φιλητᾶς': 4, 'Δρύας': 6, 'Λυκαίνιον': 2}
}


#create a dictionary for appearances of each character/name form
appearances = {}

def count_appearances(charlist, text):
    joined_text = ' '.join(text)
    for character in charlist:
        sum = joined_text.count(character)
        appearances[character] = sum

#count_appearances(final_list_chars, sections_longus)

#print(appearances)
        
#add up sums of appearances for each character
appearances = {'Νάπη': 16,
               'Διονυσοφάνης': 15,
               'Πὰν': 37,
               'Ἄστυλος': 14,
               'Γνάθων': 21,
               'Δάφνις': 408,
               'Λάμων': 74,
               'Μυρτάλη': 26,
               'Χλόη': 283,
               'Φιλητᾶς': 38,
               'Νύμφαι': 71,
               'Δρύας': 80,
               'Λυκαίνιον': 9,
               'Δόρκων': 31
               }

#create node for each character
d_and_c = nx.Graph()
for char in dict_of_dicts.keys():
    if appearances[char] > 0:
        d_and_c.add_node(char, size=appearances[char], color='cadetblue')

#add edge for each co-occurrence
for char in dict_of_dicts.keys():
    for co_char in dict_of_dicts[char].keys():
        if dict_of_dicts[char][co_char] > 0:
            d_and_c.add_edge(char, co_char, weight=dict_of_dicts[char][co_char], label='hej')

#get position for the nodes
pos_ = nx.spring_layout(d_and_c, seed=100)

#test from here:

def make_edge(x, y, text, width):
    
    '''Creates a scatter trace for the edge between x's and y's with given width

    Parameters
    ----------
    x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])
    
    y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])
    
    width: the width of the line

    Returns
    -------
    An edge trace that goes between x0 and x1 with specified width.
    '''
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'darkblue'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')
    


# For each edge, make an edge_trace, append to list
edge_trace = []
for edge in d_and_c.edges():
    
    if d_and_c.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]

        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]
        
        text   = str(d_and_c.edges()[edge]['weight'])
        
        trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                           0.15*d_and_c.edges()[edge]['weight'])

        edge_trace.append(trace)
    
# Make a node trace
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "middle center",
                        textfont_size = 28,
                        textfont_color = 'purple',
                        mode      = 'markers+text',
                        hoverinfo = 'none',
                        fillcolor='lightgreen',
                        marker = dict(color=[], size=[])
)

# For each node get the position and size and add to the node_trace
for node in d_and_c.nodes():
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['text'] += tuple(['<b>' + node + '</b> ']) 
    node_trace['marker']['size'] += tuple([d_and_c.nodes()[node]['size']*0.5])
    node_trace['marker']['color'] += tuple([d_and_c.nodes()[node]['color']])


layout = go.Layout(
    paper_bgcolor='rgba(0,200,100,0)',
    plot_bgcolor='rgba(0,0,100,0)'
)

fig = go.Figure(layout = layout)

for trace in edge_trace:
    fig.add_trace(trace)

fig.add_trace(node_trace)

fig.update_layout(showlegend = False)

fig.update_xaxes(showticklabels = False)

fig.update_yaxes(showticklabels = False)

fig.show()



#measure degree centrality
deg_cent = nx.degree_centrality(d_and_c)
for k, v in deg_cent.items():
    print(k, round(v, 2))
deg_bet = nx.betweenness_centrality(d_and_c)

#print(deg_bet)
