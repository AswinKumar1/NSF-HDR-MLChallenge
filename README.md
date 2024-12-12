# [NSF-HDR-MLChallenge](https://www.codabench.org/competitions/3223/)

### Introduction
Sea level observations from the National Data Buoy Center (NDBC) buoys along the US East Coast are essential for monitoring activities and have significant societal impacts. These buoys, equipped with advanced instruments, provide continuous real-time data on tidal variations, storm surges, and long-term sea level trends. This data is transmitted via satellite to shore-based stations, offering detailed records crucial for predicting and monitoring coastal flooding, a major threat to coastal communities. Understanding storm surges and long-term sea level rise helps in issuing timely warnings, preparing for severe weather events, and implementing coastal protection measures to combat erosion.

The information gathered from these buoys also plays a vital role in ensuring maritime safety and supporting navigation for commercial and recreational vessels. Accurate sea level data are necessary for maintaining and planning maritime infrastructure, such as harbors and ports. Additionally, long-term sea level records are invaluable for studying climate change impacts, providing evidence of global warming effects like melting polar ice and thermal expansion of seawater. These observations support the resilience and sustainability of coastal economies, which rely heavily on tourism, fishing, and shipping industries. Moreover, real-time data enable quick response to emergencies, such as tsunamis or storm surges, thus reducing the risk to human lives and property and ensuring the safety and sustainability of coastal regions along the US East Coast.

Predicting sea level anomaly events, such as extreme storm surges or unusually high tides, is challenging along the low-lying US east coast region due to the complex interplay of atmospheric, oceanic, and climatic factors. These events are influenced by a combination of wind patterns, atmospheric pressure changes, and ocean currents, making accurate forecasting difficult with traditional methods. However, modern machine learning tools have the potential to enhance prediction capabilities by analyzing vast datasets from various sources, identifying patterns, and improving the accuracy of forecasts. Machine learning algorithms can process real-time data from NDBC buoys, historical sea level records, and meteorological information to provide more reliable predictions of extreme sea level events, thereby improving preparedness and mitigating the impacts on coastal communities.

### Problem Setting:
The iHARP anomaly prediction and detection challenge aims to predict anomalous sea-level observations from daily tide gauge data along the US East Coast affected by changes in the sea-level elevation values on the Atlantic Ocean. Anomalies in sea-level data can be caused by various weather and climate variability events, such as storm surges caused by hurricanes, mid-latitude storms, and tsunamis, or long-timescale weather and climate variability such as El Niño Southern Oscillation (ENSO). The ability to predict sea-level rise and detect anomalies has significant implications for coastal communities, aiding in the preparation for and mitigation of flood risks. Moreover, this challenge encourages the development of innovative approaches and solutions that can enhance our understanding of coastal processes and improve the accuracy of sea-level forecasts.

### Task:
The challenge provides a unique opportunity for participants to explore and apply advanced machine learning methodologies in a real-world context, focusing on the intricacies of sea-level dynamics and anomaly detection.

This challenge leverages a comprehensive training dataset spanning 20 years, consisting of daily sea level measurements from 12 coastal stations along the US East Coast and sea-level elevation values in the North Atlantic. Participants are tasked with predicting dates of sea-level anomalies at various stations on the US East Coast.

### Goal:
The goal is to enable participants to develop models that can accurately predict sea level anomalies at coastal stations for the next 10 years not included in the training dataset. The participants need to predict the sea-level anomalies across the 12 stations for each day.
