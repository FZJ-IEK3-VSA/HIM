<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# HIM- Hydrogen Infrastructure Model for Python

HSC offers the functionality to calculate predefined hydrogen supply chain architectures with respect to spatial resolution for the analysis of explicit nationwide infrastructures.

## Installation and application

First, download and install [Anaconda](https://www.anaconda.com/). Then, clone a local copy of this repository to your computer with git

	git clone https://github.com/FZJ-IEK3-VSA/HSCPathways.git
	
or download it directly. Move to the folder

	cd HIM

and install the required Python environment via

	conda env create -f environment.yml 

To determine the optimal pipeline design, a mathematical optimization solver is required. [Gurobi](https://www.gurobi.com/) is used as default solver, but other optimization solvers can be used as well.

## Examples

A number of [**examples**](apps/) shows the capabilities of HIM. Either for [abstract cost analyses](apps/Example%20-%20Abstract%20analysis%20without%20geoferenced%20locations.ipynb) 
![Supply chain cost comparison](apps/results/FigureComparison.png)

or for [exact infrastructure design](apps/Example%20Hydrogen%20Supply%20Chain%20Cost%20Generation.ipynb) 
![Infrastructure design](apps/results/SupplyChain.png)


## License

MIT License

Copyright (C) 2016-2019 Markus Reuss (FZJ IEK-3), Thomas Grube (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.
If not, see https://opensource.org/licenses/MIT

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="https://www.fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2019-02-04_459x244_2480x1317.jpg?__blob=normal" alt="Abteilung VSA"></a> 

We are the [Techno-Economic Energy Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the [Forschungszentrum Jülich](www.fz-juelich.de/). Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.


## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050 – A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
