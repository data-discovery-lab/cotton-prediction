import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

class data_explore(object):
    def __init__(self, args):
        self.file = args[1]
        self.type = args[2]
        self.cols = ['FID', 'ID','FIELD','YLD00','YLD01','YLD02','YLD03','AVGYLD','ELE','SLOPE','CURV',
           'PRO','PLAN','EC_SH','EC_DP','BAND1','BAND2','BAND3','BAND4','VI00_520','VI00_528',
           'VI00_613','VI00_707','VI00_715','VI00_723','VI00_816','VI00_824','VI00_901','VI00_917',
           'VI01_616','VI01_624','VI01_710','VI01_827','VI01_912','VI02_518','VI02_526','VI02_619',
           'VI02_713','VI02_721','VI02_830','VI02_907','VI02_923','VI03_505','VI03_529','VI03_606',
           'VI03_622','VI03_724','VI03_825','VI03_926']
        self.ndvi_cols = ['VI00_520','VI00_528',
           'VI00_613','VI00_707','VI00_715','VI00_723','VI00_816','VI00_824','VI00_901','VI00_917',
           'VI01_616','VI01_624','VI01_710','VI01_827','VI01_912','VI02_518','VI02_526','VI02_619',
           'VI02_713','VI02_721','VI02_830','VI02_907','VI02_923','VI03_505','VI03_529','VI03_606',
           'VI03_622','VI03_724','VI03_825','VI03_926']
        self.band_cols = ['BAND1','BAND2','BAND3','BAND4']
        self.yld_cols = ['YLD00','YLD01','YLD02','YLD03','AVGYLD']
        self.colors = ["#0088cc", "#ff7f0e"]

    def read(self):
        return pd.read_table(self.file, delimiter=',')

    def clean(self, df):
        if self.type == 'ndvi-boxplot' or self.type == 'ndvi-time':
            ndvi = df[self.ndvi_cols]
            ndvi_stack = ndvi.stack()
            ndvi_stack_df = pd.DataFrame(ndvi_stack).reset_index()
            ndvi_stack_df.rename(columns={'level_0': 'FID', 'level_1': 'time', 0: 'NDVI'}, inplace=True)
            ndvi_final = pd.merge(ndvi_stack_df, df[['FID', 'FIELD']], on='FID', how='left')
            ndvi_final['time'] = ndvi_final['time'].apply(self.time_convert)
            ndvi_final['year'] = ndvi_final['time'].apply(lambda r: str(r.year))
            ndvi_final['month'] = ndvi_final['time'].apply(lambda r: str(r.month))
            ndvi_final['day'] = ndvi_final['time'].apply(lambda r: str(r.day))
            ndvi_final_by_month = ndvi_final.groupby(['FIELD', 'year', 'month']).NDVI.mean()
            ndvi_final_by_month_df = pd.DataFrame(ndvi_final_by_month).reset_index()
            return ndvi_final_by_month_df
        elif self.type == 'band-bar':
            band = df[self.band_cols]
            band_stack = band.stack()
            band_stack_df = pd.DataFrame(band_stack).reset_index()
            band_stack_df.rename(columns={'level_0': 'FID', 'level_1': 'band', 0: 'value'}, inplace=True)
            band_final = pd.merge(band_stack_df, df[['FID', 'FIELD']], on='FID', how='left')
            return band_final
        elif self.type == 'yield-scatter' or self.type == 'yield-stripplot':
            yld = df[self.yld_cols]
            yld_stack = yld.stack()
            yld_stack_df = pd.DataFrame(yld_stack).reset_index()
            yld_stack_df.rename(columns={'level_0': 'FID', 'level_1': 'time', 0: 'value'}, inplace=True)
            yld_final = pd.merge(yld_stack_df, df[['FID', 'FIELD']], on='FID', how='left')
            return yld_final
        elif self.type == 'yield-time':
            yld = df[self.yld_cols]
            yld_stack = yld.stack()
            yld_stack_df = pd.DataFrame(yld_stack).reset_index()
            yld_stack_df.rename(columns={'level_0': 'FID', 'level_1': 'time', 0: 'value'}, inplace=True)
            yld_final = pd.merge(yld_stack_df, df[['FID', 'FIELD']], on='FID', how='left')
            yld_final_no_avg = yld_final[yld_final['time'] != 'AVGYLD']
            yld_final_no_avg['year'] = yld_final_no_avg['time'].apply(lambda r: pd.to_datetime('20' + r[3:]).year)
            yld_final_no_avg_by_month = yld_final_no_avg.groupby(['FIELD', 'year']).value.mean()
            yld_final_no_avg_by_month = pd.DataFrame(yld_final_no_avg_by_month).reset_index()
            return yld_final_no_avg_by_month
        else:
            return df

    def draw(self, df):
        if self.type == 'ndvi-boxplot':
            sns_plot = sns.boxplot(x="NDVI", y="FIELD", data=df,
                        whis=np.inf, palette=sns.color_palette(self.colors))
            sns_plot = sns.swarmplot(x="NDVI", y="FIELD", data=df,
                          size=2, color=".3", linewidth=0)
            figure = sns_plot.get_figure()
            figure.savefig("ndvi-boxplot.png")
        elif self.type == 'ndvi-time':
            g = sns.FacetGrid(df, row="FIELD", col="year", hue='FIELD',
                              palette=sns.color_palette(self.colors), margin_titles=True)
            sns_plot = g.map(plt.scatter, "month", 'NDVI')
            sns_plot = g.map(plt.plot, "month", 'NDVI')
            sns_plot.savefig("ndvi-time.png")
        elif self.type == 'band-bar':
            g = sns.FacetGrid(df, row="band", col="FIELD", hue='FIELD', palette=sns.color_palette(self.colors),
                              margin_titles=True)
            sns_plot = g.map(plt.bar, "FID", 'value')
            sns_plot.savefig("band-bar.png")
        elif self.type == 'yield-scatter':
            g = sns.FacetGrid(df, row="FIELD", col="time", hue='FIELD', palette=sns.color_palette(self.colors),
                              margin_titles=True)
            sns_plot = g.map(plt.scatter, "FID", 'value')
            sns_plot.savefig("yield-scatter.png")
        elif self.type == 'yield-stripplot':
            f, ax = plt.subplots()
            sns.despine(bottom=True, left=True)
            sns_plot = sns.stripplot(x="value", y="time", hue="FIELD",
                          data=df, dodge=True, jitter=True, palette=sns.color_palette(self.colors),
                          alpha=.25, zorder=1)

            sns_plot = sns.pointplot(x="value", y="time", hue="FIELD",
                          data=df, dodge=.532, join=False, palette="dark",
                          markers="d", scale=.75, ci=None)

            # Improve the legend
            handles, labels = ax.get_legend_handles_labels()
            sns_plot = ax.legend(handles[0:2], labels[0:2], title="FIELD",
                      handletextpad=0, columnspacing=1,
                      loc="right", ncol=3, frameon=True)
            fig = sns_plot.get_figure()
            fig.savefig("yield-stripplot")
        elif self.type == 'yield-time':
            g = sns.FacetGrid(df, col="FIELD", hue='FIELD', palette=sns.color_palette(self.colors),
                              margin_titles=True)
            sns_plot = g.map(plt.scatter, "year", 'value')
            sns_plot = g.map(plt.plot, "year", 'value')
            sns_plot.savefig("yield-time")
        elif self.type == 'heatmap-corr':
            sns.set(style="white")

            # Compute the correlation matrix
            corr = df.corr()

            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(11, 9))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns_plot = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            fig = sns_plot.get_figure()
            fig.savefig("heatmap-corr")



    def time_convert(self, r):
        year, month_day = r.split('_')
        year = '20' + year[2:]
        month_day = month_day if len(month_day) == 4 else '0' + month_day
        month_day = month_day[:2] + '-' + month_day[2:]
        return pd.to_datetime(year + '-' + month_day)

if __name__ == '__main__':
    data_explore = data_explore(sys.argv)
    df_raw = data_explore.read()
    df_clean = data_explore.clean(df_raw)
    data_explore.draw(df_clean)