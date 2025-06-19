import unittest

from .mine_html_templates import generate_tests_from_html


class TestMineTests(unittest.TestCase):
    def test_absent_nested(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Paradigm for Nuclear Safety</title>
</head>
<body>
    <header>
        <div class="logo" aria-label="Japan Nuclear Safety Institute Logo"></div>
    </header>
    
    <main>
        <h1>New Paradigm for Nuclear Safety</h1>
        
        <div class="attribution">
            <p>Thursday, April 25, 2013</p>
            <p>Japan Nuclear Safety Institute</p>
            <p>Shojiro Matsuura, Chairman</p>
        </div>
    </main>
    
    <footer>
        <div class="footer-line"></div>
        <div class="footer-content">
            <div class="footer-text">
                <p class="tagline">In Pursuit of Improved Nuclear Safety</p>
                <p class="copyright">Copyright © 2012 by Japan Nuclear Safety Institute. All Rights Reserved.</p>
            </div>
            <div class="footer-logo" aria-label="Japan Nuclear Safety Institute Logo"></div>
        </div>
    </footer>
</body>
"""
        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertEqual(len([test for test in tests if test["type"] == "absent"]), 2)

    def test_text_basic(self):
        html_content = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bone Morphology Description</title>
</head>
<body>
    <main>
        <p>The posterior end exhibits a curved process to articulate with the angular. Aside from the process, the rest of the posterior end has slight curvatures for articulation, but is mostly uniform. Ventral border of the bone is mostly straight, with slight curvature (FIG. 20).</p>
        
        <p><span class="section-heading">Lateral</span>- A spine runs from the anterior-most tip, reduces in height ~3/4 way down toward posterior, and terminates at the center of the posterior notch. A fossa is present on the dorsal side of the spine. The posterior end exhibits more relief than in medial view, with the medial side of the posterior process extending past the posterior notch.</p>
        
        <p><span class="section-heading">Ontogeny</span>- Anterior tip is sharply pointed in AR12 and AR1 with AR2 being rounded, though this could be due to breakage. Anterior dorsal margin is straight and flat in AR12; AR2 shows little curvature and AR1 shows the most curvature; curving outward dorsally. Dorsal incisure is anteroposteriorly oriented in AR12, in AR2 there is some ventral curvature, and in AR1 there is a posteroventral curvature. Both AR1 and AR3 are curved on the ventral margin while AR12 is mostly straight. Posterior end of AR1 exhibits four undulations, ventral process is not yet extended. A fossa is present dorsal to the ventral process, not seen on AR12 or AR2. In medial view the lateral ridge is visible posteriorly in AR1 and AR2l the ridge does not fully extend anteriorly. In lateral view of the posterior the ventral process is present on AR2, but not fully extended posteriorly. Tip of the anterodorsal process is sharply pointed in AR1 and AR2, rounded in AR12. A short ridge is present on the dorsal edge of the dorsal process of AR1. The short ridge on the posterodorsal process of AR2 is slightly more ventral than in AR1. On AR12 the ridge is long and positioned most ventral. The lateral ridge is closest to the ventral margin in AR1. In AR2 the ridge is positioned more dorsally and in AR12 the ridge terminates and the anterior tip. The section of bone ventral to the lateral ridge appears to thin with age. The posterior notch on AR12 is curved anteriorly and the medial side of the notch extends posteriorly</p>
    </main>
    
    <footer>
        <p>46</p>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)
        self.assertGreater(len(tests), 5)

    def test_big_headers(self):
        html_content = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FORANE 427A Comparative Data</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 724px;
            margin: 0 auto;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        table, th, td {
            border: 1px solid black;
        }
        
        th, td {
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        .logo {
            width: 150px;
            height: 80px;
            background-color: #eee;
            border: 1px solid #000;
            margin-top: 20px;
        }
        
        footer {
            margin-top: 40px;
            font-size: 0.8em;
        }
        
        .company-info {
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
        }
        
        .contact {
            text-align: right;
        }
    </style>
</head>
<body>
    <header>
        <h1>Comparative data</h1>
    </header>
    
    <main>
        <table>
            <thead>
                <tr>
                    <th>Parameters</th>
                    <th>R-22</th>
                    <th>FORANE<sup>®</sup> 427A</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Evaporating temperature</td>
                    <td>2.7 °C</td>
                    <td>1.1 °C</td>
                </tr>
                <tr>
                    <td>Condensing temperature</td>
                    <td>40.3 °C</td>
                    <td>44.0 °C</td>
                </tr>
                <tr>
                    <td>Suction temperature</td>
                    <td>7.1 °C</td>
                    <td>9.2 °C</td>
                </tr>
                <tr>
                    <td>Suction pressure</td>
                    <td>5.4 bar</td>
                    <td>5.0 bar</td>
                </tr>
                <tr>
                    <td>Discharge temperature</td>
                    <td>69.5 °C</td>
                    <td>71.1 °C</td>
                </tr>
                <tr>
                    <td>Discharge pressure</td>
                    <td>15.5 bar</td>
                    <td>17.1 bar</td>
                </tr>
                <tr>
                    <td>Cooling power</td>
                    <td>431 KW</td>
                    <td>376 KW</td>
                </tr>
                <tr>
                    <td>Power consumption</td>
                    <td>122 kW</td>
                    <td>124 kW</td>
                </tr>
                <tr>
                    <td>Residual mineral oil</td>
                    <td>-</td>
                    <td>11%</td>
                </tr>
            </tbody>
        </table>
        
        <p>During this field test, very satisfactory running conditions were reached immediately. The temperature set points were easily achieved with similar energy consumption as compared to R-22 despite a high level of residual mineral oil in the circuit. The performance of the installation continues to satisfy the customer's requirements after more than one year of service.</p>
        
        <p>FORANE<sup>®</sup> 427A consequently fully satisfies the requirements of the European regulations while enabling existing equipment to continue to perform well without the need for any long and costly plant modifications.</p>
        
        <p>The versatility of FORANE<sup>®</sup> 427A is also appreciated as it can be used to retrofit low temperature refrigeration equipment as well as air-conditioning installations, resulting in only one retrofit refrigerant for all R-22 units.</p>
        
        <p>Combining environmental friendliness, high performance and simplicity is today a reality with FORANE<sup>®</sup> 427A !</p>
    </main>
    
    <footer>
        <p>The information contained in this document is based on trials carried out by our Research Centres and data selected from the literature, but shall in no event be held to constitute or imply any warranty, undertaking, express or implied commitment from our part. Our formal specifications define the limit of our commitment. No liability whatsoever can be accepted by Arkema with regard to the handling, processing or use of the product or products concerned which must in all cases be employed in accordance with all relevant laws and/or regulations in force in the country or countries concerned.</p>
        
        <div class="company-info">
            <div class="address">
                <div class="logo"></div>
                <p>ARKEMA<br>
                420 rue d'Estienne d'Orves<br>
                92700 Colombes - France<br>
                www.arkema.com</p>
            </div>
            
            <div class="contact">
                <p>www.forane.com / info.forane@arkema.com</p>
            </div>
        </div>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertFalse(any(test for test in tests if test["type"] == "absent" and "Comparative data" in test["text"]))

    def test_page_num(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Paper - Page 47</title>
</head>
<body>
    <main>
        <div class="image" aria-label="Bar Plot for supportprotest comparing data from 2020 and 2023, showing different levels of support from 'Don't know/No answer' to 'Strongly disagree'"></div>
        
        <p class="figure-caption">Figure 4.3: The COVID-19 pandemic resulted in meaningful increase in the support for other groups' protests among Panamanians.</p>
        
        <section>
            <h2>4.2.2 Demographically-Informed Opinion Assignment</h2>
            
            <p>Our model does not endow opinions randomly; instead, we leverage data to assign activists in a more realistic fashion. We use Latinobarómetro survey data from 2020 and 2023, both of which contain the three measurements of support for protest. Then, we explored which demographic groups were more likely to be activists; these are young adults and individuals at either extreme of the financial spectrum. We use this insight to influence the assignment of opinions: our logistic equations make it so that individuals with these characteristics are more likely to be labeled as activists as the probabilistic endowment happens. The code ensures that the proportion of activists overall remains exactly as desired and that there are activists who do not belong to these identified groups</p>
        </section>
        
        <section>
            <h2>4.2.3 Identity Factored into Social Influence</h2>
            
            <p>The similarity formula for Panama is built as follows, taking in nine demographic factors stored as node attributes. These are gender, age, nationality, financial status, highest level of education, level of employment, geographical region, party affiliation, and ethnicity (respectively encoded as gend, age, nation, fin, edu, emp, region, paff, and ethni). Each one of these factors has an associated weight; in this model, all factors were weighted as 0.10, except for level of education and financial status which received 0.15. Our code establishes logical rules to compare the two individuals on each dimension and return a factor by which to multiply the weight. These factors can be absolute or relative, based on the demographic dimension in question. For example, the logical conditions for gender returns 1 if same or 0 if different, while age returns a float value between 0 and 1 according to how close in age the individuals are. Once the pairwise similarity</p>
        </section>
    </main>
    
    <footer>
        <p>47</p>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertEqual(len([test for test in tests if test["type"] == "absent"]), 1)

    def test_div_footer(self):
        html_content = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Being Thai: A Narrow Identity in a Wide World</title>
    <style>
        body {
            font-family: Times New Roman, serif;
            line-height: 1.5;
            max-width: 710px;
            margin: 0 auto;
            padding: 20px;
        }
        .color-bars {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .left-bar, .right-bar {
            height: 20px;
            width: 200px;
            border: 1px solid #000;
        }
        .left-bar {
            background: linear-gradient(to right, #000, #fff);
        }
        .right-bar {
            background: linear-gradient(to right, yellow, magenta, cyan, green, blue, red, black, yellow, pink, lightblue);
        }
        .page-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        .footnote {
            font-size: 0.8em;
            vertical-align: super;
        }
        ol {
            margin-left: 20px;
        }
        .page-footer {
            display: flex;
            justify-content: space-between;
            margin-top: 40px;
            font-size: 0.8em;
            color: #666;
        }
        .registration-mark {
            font-size: 1.2em;
            color: #000;
        }
    </style>
</head>
<body>
    <div class="color-bars">
        <div class="left-bar"></div>
        <span class="registration-mark">⊕</span>
        <div class="right-bar"></div>
    </div>

    <div class="page-header">
        <div>Being Thai: A Narrow Identity in a Wide World</div>
        <div>333</div>
    </div>

    <p>hard to create a political and cultural narrative that appeals to old ideas about being Thai at this moment of perceived vulnerability.</p>

    <h1>The Concept of "Thainess"</h1>

    <p>Thainess is a political notion that originally evolved to support an authoritarian government and was then re-shaped for periods of more democratic rule.<span class="footnote">13</span> Thailand has long oscillated between dictatorship and democracy and, in either case, a sense of the "Thai style" (<em>baeb Thai</em>) is commonly invoked. Under these conditions a military coup may be thought to "advance Thai-style democracy".<span class="footnote">14</span> This is obviously fraught with difficulties and requires government agencies, most notably the Ministry of Culture, to work hard on shaping national identity.<span class="footnote">15</span> Thailand's geographical and cultural diversity means that there are inevitable deviations. Some of these have been astutely handled, especially in the northeastern provinces where the Lao-speaking minority has been integrated as <em>chao isan</em>. Nowadays it is only at the margins that their "Isan-ness" remains a contested sub-category of Thainess.<span class="footnote">16</span> In earlier generations there were more explicit challenges to the suggestion of Isan as Thai.<span class="footnote">17</span> Similar defiance has emerged in both the northern provinces<span class="footnote">18</span> and in the Malay Muslim majority areas of the far south.<span class="footnote">19</span> At various times there have been suggestions, as reported by the anthropologist Nick Tapp, that "Thainess" was disintegrating.<span class="footnote">20</span> It is in response to these persistent challenges that Prayuth's military government has sought to create its own revised version of the national ideal.</p>

    <p>For the military government the codification of Thailand's core values has created new opportunities to stamp its preferred identity on society. In a key speech soon after he took power in 2014, Prayuth identified disunity as a problem in Thai society that would, in his words, "urgently require inclusive cooperation from people of all levels, gender and age".<span class="footnote">21</span> His approach was to "define clear core values of Thai people so that we can build a strong nation". These values draw on cultural ideas that have existed for many decades and have enjoyed the favour of previous military rulers. The full list of these twelve values is:</p>

    <ol>
        <li>Upholding the three main pillars of the country: the nation, the religion and the monarchy;</li>
        <li>Showing honesty, sacrifice and patience, with a positive attitude for the interest of the public;</li>
        <li>Practicing filial piety towards parents, guardians and teachers;</li>
        <li>Seeking both direct and indirect knowledge and education;</li>
    </ol>

    <div class="registration-mark" style="position: absolute; left: 10px; bottom: 50%;">⊕</div>
    <div class="registration-mark" style="position: absolute; right: 10px; bottom: 50%;">⊕</div>

    <div class="page-footer">
        <div>15-03450 10a Thailand.indd 333</div>
        <div>15/2/16 8:24 am</div>
    </div>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertEqual(len([test for test in tests if test["type"] == "absent"]), 4)

    def test_table(self):
        html_content = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distribuição da população na estrutura socioocupacional - Brasil 2000</title>
    <style>
        body {
            font-family: Times New Roman, serif;
            line-height: 1.4;
            max-width: 686px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        th, td {
            border: 1px solid black;
            padding: 2px 4px;
            text-align: center;
        }
        th {
            font-weight: bold;
        }
        .left-align {
            text-align: left;
        }
        .source {
            font-size: 0.8rem;
            font-style: italic;
            margin-top: 10px;
        }
        footer {
            margin-top: 20px;
            font-size: 0.8rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <header>
        <div></div>
        <div>Alexandre Gori Maia e Waldir José de Quadros ■ 417</div>
    </header>

    <h3>Apêndice A - Distribuição da população na estrutura socioocupacional - Brasil 2000</h3>

    <table>
        <thead>
            <tr>
                <th rowspan="2" class="left-align">Grupo Ocupacional</th>
                <th rowspan="2" class="left-align">Classe Ocupacional</th>
                <th colspan="2">Superior</th>
                <th colspan="2">Médio</th>
                <th colspan="2">Baixo</th>
                <th colspan="2">Interior</th>
                <th colspan="2">Ínfimo</th>
                <th colspan="2">Total</th>
            </tr>
            <tr>
                <th>N (1.000s)</th>
                <th>%</th>
                <th>N (1.000s)</th>
                <th>%</th>
                <th>N (1.000s)</th>
                <th>%</th>
                <th>N (1.000s)</th>
                <th>%</th>
                <th>N (1.000s)</th>
                <th>%</th>
                <th>N (1.000s)</th>
                <th>%</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="3" class="left-align">Empregadores</td>
                <td class="left-align">A-1 Empregadores (> 10)</td>
                <td>608</td>
                <td>67,3</td>
                <td>185</td>
                <td>20,4</td>
                <td>86</td>
                <td>9,6</td>
                <td>16</td>
                <td>1,8</td>
                <td>8</td>
                <td>0,9</td>
                <td>903</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">A-2 Empregadores (<= 10)</td>
                <td>1.555</td>
                <td>36,9</td>
                <td>1.107</td>
                <td>26,3</td>
                <td>1.036</td>
                <td>24,7</td>
                <td>341</td>
                <td>8,1</td>
                <td>171</td>
                <td>4,1</td>
                <td>4.213</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Total</td>
                <td>2.162</td>
                <td>42,3</td>
                <td>1.292</td>
                <td>25,3</td>
                <td>1.126</td>
                <td>22,0</td>
                <td>357</td>
                <td>7,0</td>
                <td>179</td>
                <td>3,5</td>
                <td>5.116</td>
                <td>100</td>
            </tr>
            <tr>
                <td rowspan="3" class="left-align">Profissionais</td>
                <td class="left-align">C Profissionais Autônomos</td>
                <td>1.643</td>
                <td>21,7</td>
                <td>1.513</td>
                <td>20,0</td>
                <td>2.073</td>
                <td>27,4</td>
                <td>1.225</td>
                <td>16,2</td>
                <td>1.108</td>
                <td>14,7</td>
                <td>7.562</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">D Profissionais Assalariados</td>
                <td>4.438</td>
                <td>13,3</td>
                <td>6.030</td>
                <td>18,0</td>
                <td>11.550</td>
                <td>34,5</td>
                <td>7.027</td>
                <td>21,0</td>
                <td>4.389</td>
                <td>13,1</td>
                <td>33.434</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Total</td>
                <td>6.081</td>
                <td>14,8</td>
                <td>7.543</td>
                <td>18,4</td>
                <td>13.623</td>
                <td>33,2</td>
                <td>8.252</td>
                <td>20,1</td>
                <td>5.497</td>
                <td>13,4</td>
                <td>40.995</td>
                <td>100</td>
            </tr>
            <tr>
                <td rowspan="4" class="left-align">Massa Não-Agrícola</td>
                <td class="left-align">F Trabalhadores Autônomos</td>
                <td>657</td>
                <td>3,5</td>
                <td>1.754</td>
                <td>9,2</td>
                <td>5.561</td>
                <td>29,2</td>
                <td>5.271</td>
                <td>27,7</td>
                <td>5.788</td>
                <td>30,4</td>
                <td>19.030</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">G Trabalhadores Assalariados</td>
                <td>282</td>
                <td>0,7</td>
                <td>1.657</td>
                <td>4,3</td>
                <td>10.363</td>
                <td>27,1</td>
                <td>13.002</td>
                <td>34,0</td>
                <td>12.968</td>
                <td>33,9</td>
                <td>38.272</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">I Trabalhadores Domésticos</td>
                <td>10</td>
                <td>0,1</td>
                <td>104</td>
                <td>1,6</td>
                <td>977</td>
                <td>14,7</td>
                <td>1.810</td>
                <td>27,3</td>
                <td>3.733</td>
                <td>56,3</td>
                <td>6.633</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Total</td>
                <td>948</td>
                <td>1,5</td>
                <td>3.515</td>
                <td>5,5</td>
                <td>16.901</td>
                <td>26,4</td>
                <td>20.083</td>
                <td>31,4</td>
                <td>22.489</td>
                <td>35,2</td>
                <td>63.936</td>
                <td>100</td>
            </tr>
            <tr>
                <td rowspan="4" class="left-align">Massa Agrícola</td>
                <td class="left-align">H-1 Proprietários Conta Própria</td>
                <td>188</td>
                <td>2,0</td>
                <td>364</td>
                <td>3,8</td>
                <td>1.387</td>
                <td>14,4</td>
                <td>1.889</td>
                <td>19,7</td>
                <td>5.779</td>
                <td>60,2</td>
                <td>9.608</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">H-2 Trabalhadores Autônomos</td>
                <td>5</td>
                <td>0,5</td>
                <td>14</td>
                <td>1,5</td>
                <td>72</td>
                <td>7,6</td>
                <td>152</td>
                <td>16,1</td>
                <td>703</td>
                <td>74,3</td>
                <td>946</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">H-3 Trabalhadores Assalariados</td>
                <td>17</td>
                <td>0,2</td>
                <td>58</td>
                <td>0,6</td>
                <td>794</td>
                <td>8,4</td>
                <td>2.260</td>
                <td>23,9</td>
                <td>6.322</td>
                <td>66,9</td>
                <td>9.451</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Total</td>
                <td>210</td>
                <td>1,0</td>
                <td>436</td>
                <td>2,2</td>
                <td>2.253</td>
                <td>11,3</td>
                <td>4.301</td>
                <td>21,5</td>
                <td>12.805</td>
                <td>64,0</td>
                <td>20.005</td>
                <td>100</td>
            </tr>
            <tr>
                <td rowspan="5" class="left-align">Não-remunerados</td>
                <td class="left-align">Não-remunerados Não-Agrícolas</td>
                <td>13</td>
                <td>6,8</td>
                <td>16</td>
                <td>8,1</td>
                <td>28</td>
                <td>14,0</td>
                <td>22</td>
                <td>10,9</td>
                <td>119</td>
                <td>60,2</td>
                <td>198</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Não-remunerados Agrícolas</td>
                <td>5</td>
                <td>0,1</td>
                <td>13</td>
                <td>0,3</td>
                <td>59</td>
                <td>1,6</td>
                <td>352</td>
                <td>9,4</td>
                <td>3.302</td>
                <td>88,5</td>
                <td>3.731</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Sem Ocupação Com Renda</td>
                <td>1.567</td>
                <td>6,0</td>
                <td>2.330</td>
                <td>8,9</td>
                <td>5.395</td>
                <td>20,7</td>
                <td>6.821</td>
                <td>26,2</td>
                <td>9.964</td>
                <td>38,2</td>
                <td>26.078</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Sem Ocupação Sem Renda</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
                <td>8.094</td>
                <td>100</td>
                <td>8.094</td>
                <td>100</td>
            </tr>
            <tr>
                <td class="left-align">Ignorados</td>
                <td>177</td>
                <td>10,3</td>
                <td>202</td>
                <td>11,8</td>
                <td>364</td>
                <td>21,1</td>
                <td>337</td>
                <td>19,6</td>
                <td>640</td>
                <td>37,2</td>
                <td>1.720</td>
                <td>100</td>
            </tr>
        </tbody>
    </table>

    <p class="source">Fonte: Censo Demográfico 2000, microdados. IBGE. Elaboração dos autores.</p>

    <footer>
        RESR, Piracicaba, SP, vol. 47, nº 02, p. 389-418, abr/jun 2009 – Impressa em julho 2009
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertTrue(len(tests) > 10)

    def test_sup(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>A ROSE BY ANY OTHER NAME</title>
    <style>
        body {
            font-family: Georgia, serif;
            line-height: 1.5;
            margin: 0 auto;
            max-width: 666px;
            padding: 20px;
        }
        header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .page-number-left {
            text-align: left;
        }
        .title {
            text-align: center;
            font-weight: bold;
            flex-grow: 1;
        }
        .page-number-right {
            text-align: right;
        }
        .section-heading {
            text-align: center;
            margin: 20px 0;
        }
        p {
            text-indent: 2em;
            margin: 0 0 10px 0;
        }
        .footnotes {
            margin-top: 30px;
            border-top: 1px solid #ccc;
            padding-top: 10px;
            font-size: 0.9em;
        }
        .footnote {
            text-indent: -1.5em;
            padding-left: 1.5em;
            margin-bottom: 5px;
        }
        .italic {
            font-style: italic;
        }
        sup {
            font-size: 0.7em;
            vertical-align: super;
        }
    </style>
</head>
<body>
    <header>
        <div class="page-number-left">2016]</div>
        <div class="title">A ROSE BY ANY OTHER NAME</div>
        <div class="page-number-right">1083</div>
    </header>

    <main>
        <p>cases were decided within a year of each other (2000 and 2001, respectively). <span class="italic">Save the Manatee Club</span> largely consists of a truncated version of the <span class="italic">Consolidated-Tomoka</span> analysis, with minor adjustments to conform the opinion to the 1999 amendments. <span class="italic">Day Cruise</span>, on the other hand, closely analyzes the 1999 version of section 120.52(8). However, it is <span class="italic">Save the Manatee Club</span> that has come to dominate Florida court opinions on rulemaking challenges and not the more detailed <span class="italic">Day Cruise</span> analysis.<sup>78</sup> The following Sections will discuss the facts of the two cases, examine the differences between their analyses of section 120.52(8), and finally conclude with an opinion on which analysis is better to apply in section 120.52(8) rulemaking challenges.</p>

        <h2 class="section-heading">A. Southwest Florida Water Management District v. Save the Manatee Club, Inc.</h2>

        <p>After the legislature amended the APA, the First DCA analyzed the statutory language of section 120.52(8) again in <span class="italic">Southwest Florida Water Management District v. Save the Manatee Club, Inc.</span><sup>79</sup> <span class="italic">Save the Manatee Club</span> concerned the Southwest Florida Water Management District's (the "District's") authority to create exemptions to environmental resource permitting requirements.<sup>80</sup> South Shores Partners, Ltd. ("South Shores") applied "for a permit to develop a 720-acre tract of land in Southwest Hillsborough County."<sup>81</sup> As part of the development project, South Shores wanted "to build a connecting waterway between the [existing] canal system [on the property] and the [Tampa] Bay."<sup>82</sup> The Save the Manatee Club believed that the resulting increase in power boat traffic in this new waterway would "endanger the manatee and its habitat."<sup>83</sup></p>

        <p>The District has the authority to grant either a general permit or an environmental resource permit to a development project, depending on the type of project involved.<sup>84</sup> When granting an environmental resource permit, the District must consider "[t]he impact a proposed development will have on wildlife" as a factor; it does not have to do so when it grants a general permit.<sup>85</sup> The District granted South</p>
    </main>

    <footer class="footnotes">
        <div class="footnote">78. As of December 14, 2015, a search of the "Citing References" on WestLaw shows that <span class="italic">Save the Manatee Club</span> has been cited by forty court opinions. <span class="italic">Day Cruise</span>, by comparison, has been cited by fifteen court opinions. These numbers do not include citations to either case in DOAH decisions.</div>
        <div class="footnote">79. 773 So. 2d 594 (Fla. 1st DCA 2000).</div>
        <div class="footnote">80. <span class="italic">Id.</span> at 596.</div>
        <div class="footnote">81. <span class="italic">Id.</span></div>
        <div class="footnote">82. <span class="italic">Id.</span></div>
        <div class="footnote">83. <span class="italic">Id.</span></div>
        <div class="footnote">84. <span class="italic">See id.</span></div>
        <div class="footnote">85. <span class="italic">Id.</span></div>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        superscript_map = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "+": "⁺",
            "-": "⁻",
            "=": "⁼",
            "(": "⁽",
            ")": "⁾",
            "n": "ⁿ",
            "i": "ⁱ",
        }

        for test in tests:
            for sup in superscript_map.values():
                self.assertTrue(sup not in test.get("text", ""))
                self.assertTrue(sup not in test.get("before", ""))
                self.assertTrue(sup not in test.get("after", ""))

    def test_katex_autorender(self):
        """Test that KaTeX math expressions are properly auto-rendered when using the render_pdf_with_playwright function."""
        import asyncio
        import os
        import tempfile

        from ..synth.mine_html_templates import render_pdf_with_playwright

        # Create HTML with LaTeX expressions
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>KaTeX Auto-Render Test</title>
        </head>
        <body>
            <h1>Math Expressions Test</h1>
            
            <p>Inline math expression: \(E = mc^2\)</p>
            
            <p>Block math expression:</p>
            <p>\[
            \\frac{d}{dx}(x^n) = nx^{n-1}
            \]</p>
            
            <p>Another complex equation:</p>
            <p>\[
            \int_{a}^{b} f(x) \, dx = F(b) - F(a)
            \]</p>
        </body>
        </html>
        """

        # Create a temporary file to store the rendered PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            output_pdf_path = tmp_file.name

        # Render the HTML to PDF
        render_success = asyncio.run(render_pdf_with_playwright(html_content=html_content, output_pdf_path=output_pdf_path, png_width=800, png_height=600))

        # Check if rendering was successful
        self.assertTrue(render_success, "PDF rendering should succeed")

        # Verify PDF was created and has content
        self.assertTrue(os.path.exists(output_pdf_path), "PDF file should exist")
        self.assertTrue(os.path.getsize(output_pdf_path) > 0, "PDF file should have content")

        # The actual validation of KaTeX rendering would require visual inspection or text extraction,
        # but at minimum we can verify the file was created successfully

        print(output_pdf_path)

    def test_line_numbers(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Amendment Bill No. CS/CS/SB 7030</title>
</head>
<body>
    <header class="document-header">
        <div class="bill-title">HOUSE AMENDMENT</div>
        <div>Bill No. CS/CS/SB 7030, 1st Eng. (2019)</div>
    </header>

    <div class="amendment-label">Amendment No.</div>

    <div class="chamber-action">
        <div class="chamber-columns">
            <div class="senate-column">Senate</div>
            <div class="house-column">House</div>
        </div>
        <div style="text-align: center;">.</div>
    </div>

    <div class="horizontal-line"></div>

    <div class="horizontal-line"></div>

    <div class="amendment-content">
        <div>
            <span class="line-number">1</span>
            <div class="line-content">Representative Jenne offered the following:</div>
        </div>
        <div>
            <span class="line-number">2</span>
            <div class="line-content"></div>
        </div>
        <div>
            <span class="line-number">3</span>
            <div class="line-content"><strong>Amendment</strong></div>
        </div>
        <div>
            <span class="line-number">4</span>
            <div class="line-content">Remove lines 274-280 and insert:</div>
        </div>
        <div>
            <span class="line-number">5</span>
            <div class="line-content">c.3. Pass <span class="underline">an initial</span> a psychological evaluation, and</div>
        </div>
        <div>
            <span class="line-number">6</span>
            <div class="line-content"><span class="underline">subsequent yearly psychological evaluations before each school</span></div>
        </div>
        <div>
            <span class="line-number">7</span>
            <div class="line-content"><span class="underline">year, administered by a psychologist licensed under chapter 490</span></div>
        </div>
        <div>
            <span class="line-number">8</span>
            <div class="line-content">and designated by the Department of Law Enforcement and submit</div>
        </div>
        <div>
            <span class="line-number">9</span>
            <div class="line-content">the results of <span class="underline">such evaluations</span> <span class="strikethrough">the evaluation</span> to the sheriff's</div>
        </div>
        <div>
            <span class="line-number">10</span>
            <div class="line-content">office. The Department of Law Enforcement is authorized to</div>
        </div>
        <div>
            <span class="line-number">11</span>
            <div class="line-content">provide the sheriff's office with mental health and substance</div>
        </div>
        <div>
            <span class="line-number">12</span>
            <div class="line-content">abuse data for compliance with this paragraph.</div>
        </div>
    </div>

    <footer class="document-footer">
        <div>588513</div>
        <div>Approved For Filing: 4/23/2019 6:09:18 PM</div>
        <div>Page 1 of 1</div>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        for test in tests:
            if test["type"] == "order":
                self.assertTrue(len([c for c in test["before"] if not c.isdigit()]) > 0)
