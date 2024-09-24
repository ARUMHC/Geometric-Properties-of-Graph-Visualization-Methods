# Geometric-Properties-of-Graph-Visualization-Methods
Data Science Master Thesis 2024

# TODO

- zobaczyc o co chodzi z tą miarą assortavity DONE
- pomyślec jak podzielić grafy na kategorie DONE
- dodac wybieranie optymalnego resolution do leiden DONE
- przemyśleć czy stroić hiperparametry w clusteringu
- jak przedstawić wyniki na wykresach
- zrobić osobne experymenty dla algorytmów bez specyfikacji ilości communities
- zobaczyć czy jest problem w detekcji ilości klastrów
- zmienic funkcje tak żeby zwracało wszystkie wyniki zeby było można było zrobić boxploty z tego DONE
- pomysleć może żeby rozmiary sie generowały automatycznie, żeby nie były takie równe DONE


# Experiments ideas 
- zrobic kilka kategori z różnym wynikiem assortivity (idk co .2)
- podzielic je na rozmiary może? 
- smol to byłby 50 wierzchołków, z równymi i nierównymi rozmiarami
- medium to moze 300 wierzchłoków???
- large to 1000 (do konsultacji z siudemem)


# FILE STRUCTURE

graph_generatin_script -> funkcje do generowania grafow
clustering_script -> funkcje do experymentow
 full_cluster_experiment :
 
 * coducts ONE experiemnt for all (7) the layouts
 * returns : df with ARI layouts and algoriths for ONE graph

 stead_full_exeriment :

* generates k graphs and conducts FULL experiments on them
* it sums up ARIs and divides by k (average)
* returns : df

assortativity -> do archive?
number_of_clusters -> testy do wybierania optymalnego wybierania liczby klastrow
