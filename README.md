# BIOMEDICAL_COMPUTER_VISION_COURSE
Passion in Action polimi 2023

\title{The Kidney and Kidney Tumor Segmentation Challenge}
\author{Pucci Valentina}

\documentclass[a4paper,12pt]{article}

\usepackage{graphicx}
\graphicspath{ {2d} }
\begin{document}
\begin{center}


\LARGE \textbf \textit{\uppercase{"BIOMEDICAL COMPUTER VISION COURSE"}}

\hspace{10cm} 

\large The Kidney and Kidney Tumor Segmentation Challenge\\

\hspace{20cm} 


\normalsize Pucci Valentina\\
\normalsize Politecnico di Milano, 06/06/2023\\

\hspace{20cm} 

\hspace{20cm} 

\includegraphics[width=0.4\linewidth]{img.png}
\includegraphics[width=0.4\linewidth]{segm.png}
\includegraphics[width=0.4\linewidth]{img2.png}
\includegraphics[width=0.4\linewidth]{segm2.png}


\end{center}

\pagebreak


\LARGE \textbf{Sommario}

\begin{enumerate}
\large
\setlength\itemsep{0.2em}
	\item Introduzione
	\item Preprocessing
	\item Rete Neurale
	\item Risultati
	\item Conclusioni

\end{enumerate}

\pagebreak

\normalsize 
\section{Introduzione}
Non possedendo basi di conoscenze solide sul funzionamento delle reti neurali per la segmentazione,  la rete creata è un riadattamento di quella fornita a laboratorio durante la lezione 7. 


\section{Preprocessing}
Nella fase di preprocessing ho convertito immagini e segmentazioni da formato nifty(3D) a numpy(2D).\\
II primo passo è stato ridimensionarle per riuscire ad avere immmagini \\confrontabili tra loro, impostando una dimensione di 256x256 e 256 slices per ogni immagine 3D.
Il salvataggio è stato realizzato di slice in slice identificando le 150 centrali, dopo aver verificato tramite ITK-SNAP che in media fossero quelle contenenti la maggior parte delle informazioni utili.  In particolare sono state scelte le slices [50,200] per evitare di avere un numero troppo grande di immagini rappresentanti zone dove non sono presenti reni,  e di conseguenza segmentazioni totalmente nere.
Il programma successivamente le suddivide in set di training e set di validation (23500 e 7850 rispettivamente) numerandole in ordine crescente come (numero).npy.


\section{Rete Neurale}
La rete è stata realizzata sul modulo nn di Torch.\\
La procedura principale dell'algoritmo chiama funzioni esterne che gestiscono il loading dei dati e la realizzazione del modello,  tutto in uno stesso programma. 
Le procedure secondarie sono: "preprocessing()",  che effettua caricamento, resizing e salvataggio delle immagini,  e "My-Neural-Network()" che realizza il modello della rete neurale e si occupa di eseguire tutte le fasi di training e validation.  \\Il modello è stato realizzato sulla base della rete resnet101, utilizzata come pre-training,  riportandola a rete per segmentazione tramite convoluzione.
In merito al calcolo iterativo della loss ho scelto di utilizzare HuberLoss poichè maggiormanete si adattava al database,  con Adagrad come ottimizzatore. 
 \\Le metriche di accuratezza sono calcolate con f1score. 
\\Per tenere traccia dei risultati di loss e score è stato realizzato il file log.csv [figura2]


\section{Risultati}
I dati sotto riportati mostrano un lungo training della rete, basato su 10 epoche con learnng rate 0.001 e batch size 100 sia per immagini che segmentazioni.
I valori di loss e validation sono molto bassi e hanno un andamento simile, non sembrano essere quindi presenti fenomeni di underfitting o overfitting.



 \hspace{2cm} 

 \begin{center}
\includegraphics[width=0.8\linewidth]{log_save.png}\\
\tiny[figura2, File log.csv]

 \hspace{20cm} 
 \hspace{20cm} 

\includegraphics[width=0.8\linewidth]{graph.png}\\
\tiny[figura3,  Grafico delle loss]
 \end{center}


 \section{Conclusioni}
Il mio progetto non è ottimale ma sono soddisfatta delle nuove conoscenze acquisite e continuerò a lavorarci su per realizzare l'obiettivo della challenge. 



\end{document}
