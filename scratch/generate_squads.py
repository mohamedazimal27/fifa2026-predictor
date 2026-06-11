import csv

squads_data = []

# Helper to add players to squads
def add_squad(team, year, players):
    for name, club, age in players:
        squads_data.append({
            'team': team,
            'year': int(year),
            'player_name': name,
            'club': club,
            'age': int(age)
        })

# Spain 2010 (Actual 23-man squad details)
spain_2010 = [
    ("Iker Casillas", "Real Madrid", 29),
    ("Víctor Valdés", "Barcelona", 28),
    ("Pepe Reina", "Liverpool", 27),
    ("Gerard Piqué", "Barcelona", 23),
    ("Carles Puyol", "Barcelona", 32),
    ("Carlos Marchena", "Villarreal", 30),
    ("Joan Capdevila", "Villarreal", 32),
    ("Sergio Ramos", "Real Madrid", 24),
    ("Álvaro Arbeloa", "Real Madrid", 27),
    ("Raúl Albiol", "Real Madrid", 24),
    ("Andrés Iniesta", "Barcelona", 26),
    ("Xavi Hernández", "Barcelona", 30),
    ("Cesc Fàbregas", "Arsenal", 23),
    ("Xabi Alonso", "Real Madrid", 28),
    ("Sergio Busquets", "Barcelona", 21),
    ("Javi Martínez", "Athletic Bilbao", 21),
    ("David Silva", "Valencia", 24),
    ("Juan Mata", "Valencia", 22),
    ("David Villa", "Barcelona", 28),
    ("Fernando Torres", "Liverpool", 26),
    ("Pedro Rodríguez", "Barcelona", 22),
    ("Fernando Llorente", "Athletic Bilbao", 25),
    ("Jesús Navas", "Sevilla", 24)
]
add_squad("Spain", 2010, spain_2010)

# Germany 2014 (Actual 23-man squad details)
germany_2014 = [
    ("Manuel Neuer", "Bayern Munich", 28),
    ("Roman Weidenfeller", "Borussia Dortmund", 33),
    ("Ron-Robert Zieler", "Hannover 96", 25),
    ("Kevin Großkreutz", "Borussia Dortmund", 25),
    ("Matthias Ginter", "Freiburg", 20),
    ("Benedikt Höwedes", "Schalke 04", 26),
    ("Mats Hummels", "Borussia Dortmund", 25),
    ("Erik Durm", "Borussia Dortmund", 22),
    ("Philipp Lahm", "Bayern Munich", 30),
    ("Per Mertesacker", "Arsenal", 29),
    ("Jérôme Boateng", "Bayern Munich", 25),
    ("Shkodran Mustafi", "Sampdoria", 22),
    ("Sami Khedira", "Real Madrid", 27),
    ("Bastian Schweinsteiger", "Bayern Munich", 29),
    ("Mesut Özil", "Arsenal", 25),
    ("Julian Draxler", "Schalke 04", 20),
    ("Toni Kroos", "Bayern Munich", 24),
    ("Mario Götze", "Bayern Munich", 22),
    ("Christoph Kramer", "Borussia Mönchengladbach", 23),
    ("Thomas Müller", "Bayern Munich", 24),
    ("Miroslav Klose", "Lazio", 36),
    ("Lukas Podolski", "Arsenal", 29),
    ("André Schürrle", "Chelsea", 23)
]
add_squad("Germany", 2014, germany_2014)

# France 2018 (Actual 23-man squad details)
france_2018 = [
    ("Hugo Lloris", "Tottenham", 31),
    ("Steve Mandanda", "Marseille", 33),
    ("Alphonse Areola", "PSG", 25),
    ("Benjamin Pavard", "Stuttgart", 22),
    ("Presnel Kimpembe", "PSG", 22),
    ("Raphaël Varane", "Real Madrid", 25),
    ("Samuel Umtiti", "Barcelona", 24),
    ("Adil Rami", "Marseille", 32),
    ("Djibril Sidibé", "Monaco", 25),
    ("Benjamin Mendy", "Manchester City", 23),
    ("Lucas Hernandez", "Atletico Madrid", 22),
    ("Paul Pogba", "Manchester United", 25),
    ("Thomas Lemar", "Monaco", 22),
    ("Corentin Tolisso", "Bayern Munich", 23),
    ("N'Golo Kanté", "Chelsea", 27),
    ("Blaise Matuidi", "Juventus", 31),
    ("Steven Nzonzi", "Sevilla", 29),
    ("Antoine Griezmann", "Atletico Madrid", 27),
    ("Olivier Giroud", "Chelsea", 31),
    ("Kylian Mbappé", "PSG", 19),
    ("Ousmane Dembélé", "Barcelona", 21),
    ("Nabil Fekir", "Lyon", 24),
    ("Florian Thauvin", "Marseille", 25)
]
add_squad("France", 2018, france_2018)

# Argentina 2022 (Actual 26-man squad details)
argentina_2022 = [
    ("Franco Armani", "River Plate", 36),
    ("Gerónimo Rulli", "Villarreal", 30),
    ("Emiliano Martínez", "Aston Villa", 30),
    ("Juan Foyth", "Villarreal", 24),
    ("Nicolas Tagliafico", "Lyon", 30),
    ("Gonzalo Montiel", "Sevilla", 25),
    ("Germán Pezzella", "Real Betis", 31),
    ("Marcos Acuña", "Sevilla", 31),
    ("Cristian Romero", "Tottenham", 24),
    ("Nicolás Otamendi", "Benfica", 34),
    ("Lisandro Martínez", "Manchester United", 24),
    ("Nahuel Molina", "Atletico Madrid", 24),
    ("Leandro Paredes", "Juventus", 28),
    ("Rodrigo De Paul", "Atletico Madrid", 28),
    ("Marcos Acuña", "Sevilla", 31),
    ("Alexis Mac Allister", "Brighton", 23),
    ("Ángel Di María", "Juventus", 34),
    ("Guido Rodríguez", "Real Betis", 28),
    ("Alejandro Gómez", "Sevilla", 34),
    ("Enzo Fernández", "Benfica", 21),
    ("Exequiel Palacios", "Leverkusen", 24),
    ("Julian Álvarez", "Manchester City", 22),
    ("Lionel Messi", "PSG", 35),
    ("Thiago Almada", "Atlanta United", 21),
    ("Ángel Correa", "Atletico Madrid", 27),
    ("Paulo Dybala", "Roma", 29),
    ("Lautaro Martínez", "Inter Milan", 25)
]
add_squad("Argentina", 2022, argentina_2022)

# France 2026 (Estimated / Future Roster)
france_2026 = [
    ("Mike Maignan", "AC Milan", 30),
    ("Brice Samba", "Lens", 32),
    ("Alphonse Areola", "West Ham", 33),
    ("William Saliba", "Arsenal", 25),
    ("Ibrahima Konaté", "Liverpool", 27),
    ("Dayot Upamecano", "Bayern Munich", 27),
    ("Jules Koundé", "Barcelona", 27),
    ("Benjamin Pavard", "Inter Milan", 30),
    ("Theo Hernandez", "AC Milan", 28),
    ("Ferland Mendy", "Real Madrid", 31),
    ("Jonathan Clauss", "Marseille", 33),
    ("Aurélien Tchouaméni", "Real Madrid", 26),
    ("Eduardo Camavinga", "Real Madrid", 23),
    ("Adrien Rabiot", "Juventus", 31),
    ("Youssouf Fofana", "Monaco", 27),
    ("Warren Zaïre-Emery", "PSG", 20),
    ("Antoine Griezmann", "Atletico Madrid", 35),
    ("Kylian Mbappé", "Real Madrid", 27),
    ("Ousmane Dembélé", "PSG", 29),
    ("Marcus Thuram", "Inter Milan", 28),
    ("Olivier Giroud", "LAFC", 39),
    ("Randal Kolo Muani", "PSG", 27),
    ("Kingsley Coman", "Bayern Munich", 30),
    ("Bradley Barcola", "PSG", 23)
]
add_squad("France", 2026, france_2026)

# Spain 2026 (Estimated / Future Roster)
spain_2026 = [
    ("Unai Simón", "Athletic Bilbao", 28),
    ("David Raya", "Arsenal", 30),
    ("Alex Remiro", "Real Sociedad", 31),
    ("Robin Le Normand", "Atletico Madrid", 29),
    ("Aymeric Laporte", "Al-Nassr", 32),
    ("Daniel Carvajal", "Real Madrid", 34),
    ("Jesús Navas", "Sevilla", 40),
    ("Alejandro Grimaldo", "Leverkusen", 30),
    ("Marc Cucurella", "Chelsea", 27),
    ("Dani Vivian", "Athletic Bilbao", 26),
    ("Rodri Hernández", "Manchester City", 29),
    ("Martín Zubimendi", "Real Sociedad", 27),
    ("Mikel Merino", "Arsenal", 29),
    ("Pedri González", "Barcelona", 23),
    ("Fermín López", "Barcelona", 23),
    ("Alex Baena", "Villarreal", 24),
    ("Dani Olmo", "Barcelona", 28),
    ("Lamine Yamal", "Barcelona", 18),
    ("Nico Williams", "Athletic Bilbao", 23),
    ("Álvaro Morata", "AC Milan", 33),
    ("Joselu Mato", "Al-Gharafa", 36),
    ("Ferran Torres", "Barcelona", 26),
    ("Mikel Oyarzabal", "Real Sociedad", 29),
    ("Ayoze Pérez", "Villarreal", 32)
]
add_squad("Spain", 2026, spain_2026)

# Argentina 2026 (Estimated / Future Roster)
argentina_2026 = [
    ("Emiliano Martínez", "Aston Villa", 33),
    ("Gerónimo Rulli", "Marseille", 34),
    ("Walter Benítez", "PSV Eindhoven", 33),
    ("Cristian Romero", "Tottenham", 28),
    ("Lisandro Martínez", "Manchester United", 28),
    ("Nicolás Otamendi", "Benfica", 38),
    ("Nahuel Molina", "Atletico Madrid", 28),
    ("Gonzalo Montiel", "Sevilla", 29),
    ("Nicolás Tagliafico", "Lyon", 33),
    ("Marcos Acuña", "River Plate", 34),
    ("German Pezzella", "River Plate", 34),
    ("Rodrigo De Paul", "Atletico Madrid", 32),
    ("Enzo Fernández", "Chelsea", 25),
    ("Alexis Mac Allister", "Liverpool", 27),
    ("Leandro Paredes", "Roma", 31),
    ("Giovani Lo Celso", "Betis", 30),
    ("Exequiel Palacios", "Leverkusen", 27),
    ("Guido Rodríguez", "West Ham", 32),
    ("Lionel Messi", "Inter Miami", 38),
    ("Ángel Di María", "Benfica", 38),
    ("Lautaro Martínez", "Inter Milan", 28),
    ("Julián Álvarez", "Atletico Madrid", 26),
    ("Alejandro Garnacho", "Manchester United", 21),
    ("Nicolás González", "Juventus", 28),
    ("Valentin Carboni", "Marseille", 21)
]
add_squad("Argentina", 2026, argentina_2026)

# Germany 2026 (Estimated / Future Roster)
germany_2026 = [
    ("Marc-André ter Stegen", "Barcelona", 34),
    ("Oliver Baumann", "Hoffenheim", 36),
    ("Alexander Nübel", "Stuttgart", 29),
    ("Antonio Rüdiger", "Real Madrid", 33),
    ("Jonathan Tah", "Leverkusen", 30),
    ("Nico Schlotterbeck", "Borussia Dortmund", 26),
    ("Waldemar Anton", "Borussia Dortmund", 29),
    ("Maximilian Mittelstädt", "Stuttgart", 29),
    ("David Raum", "RB Leipzig", 28),
    ("Benjamin Henrichs", "RB Leipzig", 29),
    ("Joshua Kimmich", "Bayern Munich", 31),
    ("Robert Andrich", "Leverkusen", 31),
    ("Pascal Groß", "Borussia Dortmund", 34),
    ("Aleksandar Pavlović", "Bayern Munich", 22),
    ("Emre Can", "Borussia Dortmund", 32),
    ("Florian Wirtz", "Leverkusen", 23),
    ("Jamal Musiala", "Bayern Munich", 23),
    ("Ilkay Gündogan", "Barcelona", 35),
    ("Leroy Sané", "Bayern Munich", 30),
    ("Thomas Müller", "Bayern Munich", 36),
    ("Kai Havertz", "Arsenal", 27),
    ("Niclas Füllkrug", "West Ham", 33),
    ("Deniz Undav", "Stuttgart", 29),
    ("Maximilian Beier", "Borussia Dortmund", 23)
]
add_squad("Germany", 2026, germany_2026)

# Brazil 2026 (Estimated / Future Roster)
brazil_2026 = [
    ("Alisson Becker", "Liverpool", 33),
    ("Ederson Moraes", "Manchester City", 32),
    ("Bento Krepski", "Al-Nassr", 26),
    ("Marquinhos Aoás", "PSG", 32),
    ("Gabriel Magalhães", "Arsenal", 28),
    ("Éder Militão", "Real Madrid", 28),
    ("Lucas Beraldo", "PSG", 22),
    ("Danilo da Silva", "Juventus", 34),
    ("Yan Couto", "Borussia Dortmund", 24),
    ("Wendell Nascimento", "Porto", 32),
    ("Guilherme Arana", "Atlético Mineiro", 29),
    ("Bruno Guimarães", "Newcastle", 28),
    ("Douglas Luiz", "Juventus", 28),
    ("João Gomes", "Wolverhampton", 25),
    ("Lucas Paquetá", "West Ham", 28),
    ("Andreas Pereira", "Fulham", 30),
    ("Éderson dos Santos", "Atalanta", 26),
    ("Vinícius Júnior", "Real Madrid", 25),
    ("Rodrygo Goes", "Real Madrid", 25),
    ("Endrick Felipe", "Real Madrid", 19),
    ("Raphinha Dias", "Barcelona", 29),
    ("Gabriel Martinelli", "Arsenal", 25),
    ("Savinho Moreira", "Manchester City", 22),
    ("Neymar da Silva", "Al-Hilal", 34)
]
add_squad("Brazil", 2026, brazil_2026)

# Write to squads.csv
with open('data/squads.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['team', 'year', 'player_name', 'club', 'age'])
    writer.writeheader()
    writer.writerows(squads_data)

print(f"Generated data/squads.csv with {len(squads_data)} entries successfully.")
