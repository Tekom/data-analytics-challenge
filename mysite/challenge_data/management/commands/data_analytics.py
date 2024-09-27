
# Import necessary Django ORM models and modules
from challenge_data.models import Creator, Idea
from django.core.management.base import BaseCommand
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from collections import Counter
from sklearn.impute import IterativeImputer
from scipy.stats import mstats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sns.set()

class Command(BaseCommand):
    help = 'Data cleaning and data analytics'

    def handle(self, *args, **kwargs):
        data_creators = Creator.objects.all().values() #Get all creators from creator models
        data_ideas = Idea.objects.all().values() #get all idead from idea model

        dataframe_ideas = pd.DataFrame(data_ideas) #convert creator data into a dataframe
        dataframe_creator = pd.DataFrame(data_creators) #convert ideas dta into a dataframe

        cleaned_df = self.handle_missing_data(df_ideas = dataframe_ideas,
                                              df_creator = dataframe_creator) #function for handling missing data on creator_department

        df_without_outliers = self.handle_outliers(df = cleaned_df) #handling outliers for integer features
        df_with_derived_columns = self.create_derived_columns(df = df_without_outliers)#add derived columns from actual columns
        analysis = self.analysis_by_creator_department(df = df_with_derived_columns)

        self.display_results(analysis)

# Step 1: Handle Missing Data
    def handle_missing_data(self,
                            df_ideas: pd.DataFrame = None,
                            df_creator: pd.DataFrame = None) -> pd.DataFrame:

        le = LabelEncoder() #for enconding categorical features
        grouped_df = df_ideas.groupby('creator_id').sum() #group idea dataset by creator_id for having each creator with his department

        df_creator['department num'] = le.fit_transform(df_creator['department']) #convert department feature into numerical
        id_deparment = dict(zip(df_creator['id'].to_list(), df_creator['department num'].to_list())) #pointing id of creator with department num

        grouped_df['department num'] = grouped_df.index.map(id_deparment) #add deparment num feature into gropuped df
        grouped_df['department num'] = grouped_df['department num'].apply(lambda x: np.nan if x == 0 else x) #replace the value of 0 with np.nan due those values are the missing values we want to find

        train_columns = ['votes', 'comments', 'views', 'department num'] #train columns for imputer
        data_imputed = self.model(train_data = grouped_df[train_columns]) #model to find missing values of creator_department feautre

        department_imputed = list(le.inverse_transform(data_imputed['department num'].astype('int'))) #transform deparment num with is corresponding label of creator_department
        grouped_df['department'] = department_imputed

        id_department = dict(zip(grouped_df.index.to_list(), grouped_df['department'].to_list()))
        df_creator['department'] = df_creator['id'].map(id_department)
        df_creator.drop('department num', axis=1, inplace=True) #drop deparment columns
        df_creator.columns = ['creator_id', 'department'] #change column names for merging

        merged_df = pd.merge(left = df_ideas, right = df_creator, how = 'left', on = 'creator_id') #merge creator and idea dataset for futher analysis
        cleaned_df = self.clean_keywords(df = merged_df) #clean keywords featture

        # self.replace_missing_data(df_creators=df_creator,
        #                           df_ideas=merged_df)

        return cleaned_df

    def replace_missing_data(self, 
                             df_creators: pd.DataFrame = None,
                             df_ideas: pd.DataFrame = None) -> None:
        
        creators_to_update = []
        ideas_to_update = []

        for index, row in df_creators.iterrows():
            try:
                # Obtener la instancia del modelo Creator
                creator = Creator.objects.get(id=row['creator_id'])
                # Asignar el nuevo valor del campo 'department'
                creator.department = row['department']
                # Añadir a la lista de actualizaciones
                creators_to_update.append(creator)
            except Creator.DoesNotExist:
                print(f"Creator with id {row['id']} not found")

        for index, row in df_ideas.iterrows():
            try:
                # Obtener la instancia del modelo Idea
                idea = Idea.objects.get(id=row['id'])
                # Asignar el nuevo valor del campo 'keywords'
                idea.keywords = row['keywords']
                # Añadir a la lista de actualizaciones
                ideas_to_update.append(idea)
            except Idea.DoesNotExist:
                print(f"Idea with id {row['id']} not found")
        
        Idea.objects.bulk_update(ideas_to_update, ['keywords'])
        Creator.objects.bulk_update(creators_to_update, ['department'])

    def clean_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        df['keywords'] = df['keywords'].apply(lambda x: x + " " if x != " " else x) #add a space beetween each value for them split them correctly
        grouped_df_department = df.groupby('department').sum(numeric_only=False) #group by department to see all the keywords related to a department
        grouped_df_department['keywords'] = grouped_df_department['keywords'].apply(lambda x: x.split(" ")) #making a list with all the keywords per department
        grouped_df_department


        departments = list(grouped_df_department.index)
        most_frequent_keywords = dict()

        for department in departments:
            test = grouped_df_department[grouped_df_department.index == department]['keywords'].to_list()[0] #take all the keywords from department
            new_keywords = []

            for keyword in test:
                if keyword == '':
                    pass
                else:
                    new_keywords.append(keyword.strip().replace(",",""))

            cantidad = dict(Counter(new_keywords))
            diccionario_ordenado = dict(sorted(cantidad.items(), key=lambda item: item[1], reverse=True)) #order dictionary from asc to des for getting the most frequent words per department
            words = list(diccionario_ordenado.keys())[0:3]#take the most 3 frequent keywords from each department

            text = ""

            for word in words:
                text += (word + ", ")

            most_frequent_keywords[department] = text.strip()

        df['keywords'] = np.where(
            df['keywords'] == " ",  
            df['department'].map(most_frequent_keywords),   
            df['keywords']               

        ) #replace missing keywords values into the dataset

        return df

    def model(self, train_data: pd.DataFrame = None) -> pd.DataFrame:
        imputer = IterativeImputer(random_state=100) #model to impute missing values

        df_imputed = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)
        df_imputed['department num'] = df_imputed['department num'].apply(lambda x: round(x)) #round imputed columns to integers

        return df_imputed

# Step 2: Handle Outliers
    def handle_outliers(self, df: pd.DataFrame = None) -> pd.DataFrame:
        outlier_columns = ['votes', 'comments', 'views'] #columns for removing outliers

        # sns.histplot(df[outlier_columns])
        # sns.boxplot(df[['votes', 'comments', 'views']])
        # plt.show()

        """
        After looking at the distribution of all 3 features, we know all of them are right skewed,
        and because we have a small dataset, winsorization could be one of he best options. we are going
        to move the 1% of our lower data will replace with percentile 1 and 5% of our highest data will be replaced por percentile 95
        """

        for column in outlier_columns:
            df[f'{column}'] = mstats.winsorize(df[column], limits=[0.01, 0.05])

        # sns.boxplot(df[['votes', 'comments', 'views']])
        # plt.show()

        return df

# Step 3: Create Derived Columns
    def create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df['total_interactions'] = df['votes'] + df['comments'] #adding total_interactions per idea
        df['average_interactions'] = round(df['total_interactions'] / 3, 2) #average of interacions per idea
       

        df['date'] = df['submitted_date_time'].apply(lambda x: str(x).split(" ")[0])
        df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d') #adding date columns without timestamp

        df['comment_ratio_per_views'] = (df['comments'] / df['views']) #ratio of comments per views
        df['vote_ratio_per_views'] = (df['votes'] / df['views']) #ratio of votes per views
        df['interactions_ratio'] = ((df['comments'] + df['votes']) / df['views']) #ratio of overall interactions per views 

        return df

# Step 4: Analyse by Creator Department
    def analysis_by_creator_department(self, df: pd.DataFrame) -> tuple:
        df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
        df['year'] = df['date'].dt.year

        interactions_by_year = df.groupby(['department', 'year'])['average_interactions'].sum().reset_index() #take sum of average_interactions from each department per year

        ratios_by_department = df.groupby('department').agg({
            'comment_ratio_per_views': 'mean',
            'vote_ratio_per_views': 'mean',
            'interactions_ratio': 'mean',
            'views': 'sum'
        }) #looking at the highest ratios per deparment

        status_counts = df.groupby(['department', 'status']).size().reset_index(name='count') #analyse status of ideas per department

        # Calculate the total count per department
        total_counts = df.groupby('department').size().reset_index(name='total_count')

        # Merge the counts with the total counts to calculate the ratio
        status_ratios = pd.merge(status_counts, total_counts, on='department')
        status_ratios['ratio'] = status_ratios['count'] / status_ratios['total_count']

        # Display the status ratios
        status_ratios.sort_values(by=['department', 'ratio'], ascending=[True, False], inplace=True)

        return (ratios_by_department, interactions_by_year, status_ratios)
    
# Step 5: Display Results
    def display_results(self, data: tuple = None) -> None:
        ratios = data[0]
        interactions_per_year = data[1]
        status_ratios = data[2]

        ratios = ratios.sort_values('interactions_ratio', ascending=False)        
        plt.figure(figsize=(14, 10))

        # Plot comments per view
        plt.subplot(3, 1, 1)
        sns.barplot(x=ratios.index, y='comment_ratio_per_views', data=ratios)
        plt.title('Comments per View by Department')
        plt.xticks(rotation=90)

        # Plot votes per view
        plt.subplot(3, 1, 2)
        sns.barplot(x=ratios.index, y='vote_ratio_per_views', data=ratios)
        plt.title('Votes per View by Department')
        plt.xticks(rotation=90)

        # Plot (comments + votes) per view
        plt.subplot(3, 1, 3)
        sns.barplot(x=ratios.index, y='interactions_ratio', data=ratios)
        plt.title('(Comments + Votes) per View by Department')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()
        
        """
            1. The Product department has the highest ratio of comments per view and (comments + votes) per view, indicating high engagement relative to the number of views.

            2. Software Development and Customer Excellence follow closely in terms of overall engagement (comments + votes) per view. 

            3. Some departments like 'xxxxx xxxxxxx xxxx' have comments but no votes, resulting in their ratio being solely based on comments.

            4. Departments like Development and Customer Experience have relatively low ratios across all metrics, suggesting lower engagement relative to views.

            5. The Sales department has a notably higher ratio of comments to votes compared to most other departments.
        """
  
        plt.figure(figsize=(14, 8))
        sns.barplot(data=interactions_per_year, x='year', y='average_interactions', hue='department')
        plt.title('Total Interactions per Department by Year')
        plt.xlabel('Year')
        plt.ylabel('Total Interactions')
        plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        """
        1. Increased activity between 2017 and 2019: The years 2017, 2018, and 2019 seem to concentrate the majority of interactions. 
        The Product department had significant activity peaks, suggesting increased interest or engagement within this department during that period.

        2. Decline in interactions after 2019: After 2019, the total number of interactions by department seems to decrease significantly. 
        This could indicate reduced participation, changes in work dynamics, or less usage of the platform maybe due to covid-19.

        3. Greater department diversity in 2018, more departments were involved with higher interaction levels compared to other years. 
        probably because different areas within the organization were more active or collaborative during that year.

        4. Consistent activity from the 'Operations' department: While not the department with the highest interactions overall, the Operations department shows consistent activity across several years, 
        indicating more sustained participation over time, albeit with relatively moderate peaks.

        5.Low activity in specific departments, some departments, such as Sales, Marketing, and QA, have very few interactions in almost all years, 
        which might indicate these departments participate less or do not use this platform as frequently for their related activities.

        """

        plt.figure(figsize=(14, 8))
        sns.barplot(x='department', y='ratio', hue='status', data=status_ratios)
        plt.title('Status Ratios per Department')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        """
        1. Strategic focus on development stages for some departments like Product Group and Sales showing high ratios, in the "Consultation" phase could indicate that these departments are in a planning or negotiation-heavy phase. 
        This could suggest a strategic focus on evaluating or negotiating key initiatives before moving forward.

        2. Operational Efficiency in Specific Departments: The high "Completed" ratio in departments such as Operations and Customer Success Management suggests strong operational processes. 
        These departments likely have efficient workflows that allow them to move tasks through the pipeline more quickly than others.

        3. Variation in Project Maturity: The relatively high ratios of "Concept" status in the Product Team and Software Development departments suggest that these areas are more involved in early-stage project work, 
        potentially focusing on innovation, research, or new initiatives compared to other departments that are more focused on completion or evaluation.
        """


# Step 6: Visualization (Optional)
