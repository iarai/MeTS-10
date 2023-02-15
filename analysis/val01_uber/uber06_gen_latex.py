if __name__ == "__main__":
    for city in ["barcelona", "berlin", "london"]:
        print(
            f"""

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\subsection{{{city.title()}}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% {city}_density_diff.qgz
\mbox{{}}
\\nopagebreak{{}}
\\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\\textwidth]{{figures/05_val01_uber/{city}_Uber_density_diff.png}}
\caption{{Density differences of Uber and \mcswts{{}} on the historic road graph {city.title()} (8am--6pm). The color encoding shows the edge density difference, negative means higher temporal coverage of \mcswts{{}} and positive values mean higher temporal coverage..}}
%\label{{figures/speed_stats/road_graph_antwerp_2021.jpg}}
\end{{figure}}

% uber03_{city}_spatial_coverage.ipynb
\\begin{{figure}}[H]
\centering
\includegraphics[width=0.85\\textwidth]{{figures/05_val01_uber/{city}_Uber_density_diff_barplot.pdf}}
\caption{{Segment density differences Uber and \mcswts{{}} on the historic road graph {city.title()} daytime (8am--6pm, segments within \t4c bounding box only). Mean density difference by road class (\ie{{}} OSM highway attribute); positive density difference means higher temporal coverage of \mcswts{{}} and negative mean higher temporal coverage.}}
%\label{{figures/speed_stats/road_graph_antwerp_2021.jpg}}
\end{{figure}}

% % uber04_{city}_speed_differences.ipynb
\\begin{{figure}}[H]
\centering
\\begin{{subfigure}}[b]{{0.45\\textwidth}}
\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_kde_highway_non_links.pdf}}
\caption{{KDE non-link road types.}}
\end{{subfigure}}
\\begin{{subfigure}}[b]{{0.45\\textwidth}}
\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_kde_highway_links.pdf}}
\caption{{KDE link road types.}}
\end{{subfigure}}
\\begin{{subfigure}}[b]{{0.9\\textwidth}}
\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_scatter_highway.png}}
\caption{{Scatter non-link (left) and link (right) road types.}}
\end{{subfigure}}
\caption{{Kernel Distribution Estimation and Scatter Plots of speeds of \mcswts{{}} (x-axis, \\texttt{{median\\_speed\\_kph}}) and Uber (y-axis, \\texttt{{speed\\_kph\\_mean}}) on the historic road graph {city.title()} daytime (8am--6pm) on the matching data, \ie{{}} within \mcswts{{}} bounding box only and where data is available at the same time and segment, for the most important road types.}}
\end{{figure}}

%\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_histogram.pdf}}
%\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_histogram_highway.pdf}}
\\begin{{figure}}[H]
\centering
\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_speed_diff.pdf}}
\caption{{Speed differences Uber and \mcswts{{}} on the historic road graph London daytime (8am--6pm) on the matching data, \ie{{}} within \mcswts{{}} bounding box only and where data is available at the same time and segment. Mean difference by road class (OSM highway attribute). Positive speed difference means higher values in \mcswts.}}
\end{{figure}}
\\begin{{figure}}[H]
\centering
\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_median_speed_kph.pdf}}
\caption{{\mcswts{{}} speeds on the historic road graph London daytime (8am--6pm) on the matching data, \ie{{}} within \mcswts{{}} bounding box only and where data is available at the same time and segment. By road class (OSM highway attribute).}}
\end{{figure}}
%\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_speed_kph_mean.pdf}}
%\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_Uber_speed_{'kph' if city!='london' else 'mph'}_stddev.pdf}}
%\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_t4c_std_speed_kph.pdf}}
%\includegraphics[width=\\textwidth]{{figures/05_val01_uber/{city.title()}_t4c_volume.pdf}}
"""
        )
