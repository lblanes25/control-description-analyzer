#!/usr/bin/env python3
"""
Tableau Workbook Generator - Control Analyzer Dashboard

This script generates a Tableau workbook (.twb) file for the Control Analyzer QA Dashboard.
It uses the XML structure of Tableau workbooks to programmatically create the required
visualizations and dashboards.

The generated workbook will include:
1. Portfolio Overview Dashboard
2. Drilldown Dashboard

Requirements:
- Python 3.6+
- Jinja2 library for templating
- lxml for XML manipulation (optional)

Note: This generates the .twb file only. You'll need to connect it to your .hyper file in Tableau.
"""

import os
import re
import xml.dom.minidom
from datetime import datetime
import uuid
from jinja2 import Template, Environment, FileSystemLoader

# Setup Jinja2 environment
env = Environment(loader=FileSystemLoader('.'))

# Define Tableau workbook XML template
TWB_TEMPLATE = """<?xml version='1.0' encoding='utf-8' ?>
<workbook source-build="2023.1.2 (20231.23.0511.0949)" source-platform="win" version="18.1" xmlns:user="http://www.tableausoftware.com/xml/user">
  <document-format-change-manifest>
    <_.fcp.AnimationOnByDefault.true...AnimationOnByDefault />
    <IntuitiveSorting />
    <_.fcp.MarkAnimation.true...MarkAnimation />
    <_.fcp.ObjectModelEncapsulateLegacy.true...ObjectModelEncapsulateLegacy />
    <_.fcp.ObjectModelTableType.true...ObjectModelTableType />
    <_.fcp.SchemaViewerObjectModel.true...SchemaViewerObjectModel />
    <SheetIdentifierTracking />
    <WindowsPersistSimpleIdentifiers />
  </document-format-change-manifest>

  <preferences>
    <preference name='ui.encoding.shelf.height' value='24' />
    <preference name='ui.shelf.height' value='26' />
  </preferences>

  <datasources>
    <datasource caption='Control Analyzer Results' inline='true' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' version='18.1'>
      <connection class='federated'>
        <named-connections>
          <named-connection caption='Control Analyzer Results' name='hyper.{{ connection_id }}'>
            <connection class='hyper' cleaning='no' compat='no' dbname='[path-to-your-hyper-file]' filename='[path-to-your-hyper-file]' port='443' server='localhost' />
          </named-connection>
        </named-connections>
        <_.fcp.ObjectModelEncapsulateLegacy.false...relation connection='hyper.{{ connection_id }}' name='Extract' table='[Extract].[Extract]' type='table' />
        <_.fcp.ObjectModelEncapsulateLegacy.true...relation connection='hyper.{{ connection_id }}' name='Extract' table='[Extract].[Extract]' type='table' />
        <metadata-records>
          <metadata-record class='column'>
            <remote-name>Control ID</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Control ID]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Control ID</remote-alias>
            <ordinal>0</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>1000</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Control Description</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Control Description]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Control Description</remote-alias>
            <ordinal>1</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>1000</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Total Score</remote-name>
            <remote-type>5</remote-type>
            <local-name>[Total Score]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Total Score</remote-alias>
            <ordinal>2</ordinal>
            <local-type>real</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>100</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Category</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Category]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Category</remote-alias>
            <ordinal>3</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>3</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>WHO Score</remote-name>
            <remote-type>5</remote-type>
            <local-name>[WHO Score]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>WHO Score</remote-alias>
            <ordinal>4</ordinal>
            <local-type>real</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>100</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>WHAT Score</remote-name>
            <remote-type>5</remote-type>
            <local-name>[WHAT Score]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>WHAT Score</remote-alias>
            <ordinal>5</ordinal>
            <local-type>real</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>100</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>WHEN Score</remote-name>
            <remote-type>5</remote-type>
            <local-name>[WHEN Score]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>WHEN Score</remote-alias>
            <ordinal>6</ordinal>
            <local-type>real</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>100</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>WHY Score</remote-name>
            <remote-type>5</remote-type>
            <local-name>[WHY Score]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>WHY Score</remote-alias>
            <ordinal>7</ordinal>
            <local-type>real</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>100</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>ESCALATION Score</remote-name>
            <remote-type>5</remote-type>
            <local-name>[ESCALATION Score]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>ESCALATION Score</remote-alias>
            <ordinal>8</ordinal>
            <local-type>real</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>100</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Missing Elements</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Missing Elements]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Missing Elements</remote-alias>
            <ordinal>9</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>31</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Vague Terms</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Vague Terms]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Vague Terms</remote-alias>
            <ordinal>10</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>400</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Audit Leader</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Audit Leader]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Audit Leader</remote-alias>
            <ordinal>11</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>20</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Audit Entity</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Audit Entity]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Audit Entity</remote-alias>
            <ordinal>12</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>50</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Year of Last Audit</remote-name>
            <remote-type>20</remote-type>
            <local-name>[Year of Last Audit]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Year of Last Audit</remote-alias>
            <ordinal>13</ordinal>
            <local-type>integer</local-type>
            <aggregation>Sum</aggregation>
            <approx-count>10</approx-count>
            <contains-null>true</contains-null>
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
          <metadata-record class='column'>
            <remote-name>Multiple Controls</remote-name>
            <remote-type>129</remote-type>
            <local-name>[Multiple Controls]</local-name>
            <parent-name>[Extract]</parent-name>
            <remote-alias>Multiple Controls</remote-alias>
            <ordinal>14</ordinal>
            <local-type>string</local-type>
            <aggregation>Count</aggregation>
            <approx-count>2</approx-count>
            <contains-null>true</contains-null>
            <collation flag='0' name='LROOT' />
            <_.fcp.ObjectModelEncapsulateLegacy.true...object-id>[Extract.{{ object_id1 }}]</_.fcp.ObjectModelEncapsulateLegacy.true...object-id>
          </metadata-record>
        </metadata-records>
      </connection>

      <!-- Calculated fields section -->
      <column datatype='boolean' name='[Calculation_{{ calc_id1 }}]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHO&quot;)' />
        <aliases>
          <alias key='false' value='Not Missing' />
          <alias key='true' value='Missing WHO' />
        </aliases>
      </column>
      <column datatype='boolean' name='[Calculation_{{ calc_id2 }}]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHAT&quot;)' />
        <aliases>
          <alias key='false' value='Not Missing' />
          <alias key='true' value='Missing WHAT' />
        </aliases>
      </column>
      <column datatype='boolean' name='[Calculation_{{ calc_id3 }}]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHEN&quot;)' />
        <aliases>
          <alias key='false' value='Not Missing' />
          <alias key='true' value='Missing WHEN' />
        </aliases>
      </column>
      <column datatype='boolean' name='[Calculation_{{ calc_id4 }}]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHY&quot;)' />
        <aliases>
          <alias key='false' value='Not Missing' />
          <alias key='true' value='Missing WHY' />
        </aliases>
      </column>
      <column datatype='integer' name='[Calculation_{{ calc_id5 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='INT(CONTAINS([Missing Elements], &quot;WHO&quot;)) + INT(CONTAINS([Missing Elements], &quot;WHAT&quot;)) + INT(CONTAINS([Missing Elements], &quot;WHEN&quot;)) + INT(CONTAINS([Missing Elements], &quot;WHY&quot;))' />
      </column>
      <column datatype='boolean' name='[Calculation_{{ calc_id6 }}]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='[Calculation_{{ calc_id5 }}] &gt;= 2' />
        <aliases>
          <alias key='false' value='Less than 2 missing' />
          <alias key='true' value='2+ Core Elements Missing' />
        </aliases>
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id7 }}]' role='dimension' type='ordinal'>
        <calculation class='tableau' formula='CASE [Category]&#13;&#10;WHEN &quot;Excellent&quot; THEN 1&#13;&#10;WHEN &quot;Good&quot; THEN 2&#13;&#10;WHEN &quot;Needs Improvement&quot; THEN 3&#13;&#10;END' />
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id8 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='{FIXED : AVG(INT(CONTAINS([Missing Elements], &quot;WHO&quot;)))}' />
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id9 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='{FIXED : AVG(INT(CONTAINS([Missing Elements], &quot;WHAT&quot;)))}' />
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id10 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='{FIXED : AVG(INT(CONTAINS([Missing Elements], &quot;WHEN&quot;)))}' />
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id11 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='{FIXED : AVG(INT(CONTAINS([Missing Elements], &quot;WHY&quot;)))}' />
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id12 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='{FIXED : AVG(INT([Calculation_{{ calc_id6 }}]))}</calculation>
      </column>
      <column datatype='real' name='[Calculation_{{ calc_id13 }}]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='{FIXED : AVG(INT([Multiple Controls] = &quot;Yes&quot;))}</calculation>
      </column>

      <layout _.fcp.SchemaViewerObjectModel.false...dim-percentage='0.5' _.fcp.SchemaViewerObjectModel.false...measure-percentage='0.4' dim-ordering='alphabetic' measure-ordering='alphabetic' show-structure='true' />
      <semantic-values>
        <semantic-value key='[Country].[Name]' value='&quot;United States&quot;' />
      </semantic-values>
      <_.fcp.ObjectModelEncapsulateLegacy.true...object-graph>
        <objects>
          <object caption='Extract' id='Extract.{{ object_id1 }}'>
            <properties context=''>
              <relation connection='hyper.{{ connection_id }}' name='Extract' table='[Extract].[Extract]' type='table' />
            </properties>
          </object>
        </objects>
      </_.fcp.ObjectModelEncapsulateLegacy.true...object-graph>
    </datasource>
  </datasources>

  <!-- Define workbook parameters -->
  <parameters>
    <parameter name='Parameter 1' />
  </parameters>

  <!-- Define worksheets -->
  <worksheets>
    <!-- Score Distribution Histogram -->
    <worksheet name='Score Distribution'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Score Distribution</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='Control Analyzer Results' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' />
          </datasources>
          <datasource-dependencies datasource='federated.{{ datasource_id }}.{{ datasource_id2 }}'>
            <column datatype='string' name='[Category]' role='dimension' type='nominal' />
            <column datatype='string' name='[Control ID]' role='dimension' type='nominal' />
            <column datatype='real' name='[Total Score]' role='measure' type='quantitative' />
            <column-instance column='[Control ID]' derivation='Count' name='[cnt:Control ID:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
            <column-instance column='[Total Score]' derivation='None' name='[none:Total Score:qk]' pivot='key' type='quantitative' />
          </datasource-dependencies>
          <shelf-sorts>
            <shelf-sort-v2 dimension-to-sort='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' direction='ASC' is-on-innermost-dimension='true' measure-to-sort-by='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[cnt:Control ID:qk]' shelf='columns' />
          </shelf-sorts>
          <aggregation value='true' />
        </view>
        <style>
          <style-rule element='mark'>
            <encoding attr='color' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' type='palette'>
              <map to='#28a745'>
                <bucket>&quot;Excellent&quot;</bucket>
              </map>
              <map to='#dc3545'>
                <bucket>&quot;Needs Improvement&quot;</bucket>
              </map>
              <map to='#ffc107'>
                <bucket>&quot;Good&quot;</bucket>
              </map>
            </encoding>
          </style-rule>
        </style>
      </table>
      <simple-id uuid='{{ uuid1 }}' />
    </worksheet>

    <!-- Category Breakdown Donut -->
    <worksheet name='Category Breakdown'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Category Breakdown</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='Control Analyzer Results' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' />
          </datasources>
          <datasource-dependencies datasource='federated.{{ datasource_id }}.{{ datasource_id2 }}'>
            <column datatype='real' name='[Calculation_{{ calc_id7 }}]' role='dimension' type='ordinal'>
              <calculation class='tableau' formula='CASE [Category]&#13;&#10;WHEN &quot;Excellent&quot; THEN 1&#13;&#10;WHEN &quot;Good&quot; THEN 2&#13;&#10;WHEN &quot;Needs Improvement&quot; THEN 3&#13;&#10;END' />
            </column>
            <column datatype='string' name='[Category]' role='dimension' type='nominal' />
            <column datatype='string' name='[Control ID]' role='dimension' type='nominal' />
            <column-instance column='[Control ID]' derivation='Count' name='[cnt:Control ID:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Calculation_{{ calc_id7 }}]' derivation='None' name='[none:Calculation_{{ calc_id7 }}:ok]' pivot='key' type='ordinal' />
            <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style>
          <style-rule element='mark'>
            <encoding attr='color' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' type='palette'>
              <map to='#28a745'>
                <bucket>&quot;Excellent&quot;</bucket>
              </map>
              <map to='#dc3545'>
                <bucket>&quot;Needs Improvement&quot;</bucket>
              </map>
              <map to='#ffc107'>
                <bucket>&quot;Good&quot;</bucket>
              </map>
            </encoding>
          </style-rule>
        </style>
      </table>
      <simple-id uuid='{{ uuid2 }}' />
    </worksheet>

    <!-- Element Radar Chart (simplified for template, real radar requires more setup) -->
    <worksheet name='Element Radar'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Element Scores</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='Control Analyzer Results' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' />
          </datasources>
          <datasource-dependencies datasource='federated.{{ datasource_id }}.{{ datasource_id2 }}'>
            <column datatype='string' name='[Category]' role='dimension' type='nominal' />
            <column datatype='real' name='[ESCALATION Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHAT Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHEN Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHO Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHY Score]' role='measure' type='quantitative' />
            <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style />
      </table>
      <simple-id uuid='{{ uuid3 }}' />
    </worksheet>

    <!-- Missing Elements Bar Chart -->
    <worksheet name='Missing Elements'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Missing Elements</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='Control Analyzer Results' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' />
          </datasources>
          <datasource-dependencies datasource='federated.{{ datasource_id }}.{{ datasource_id2 }}'>
            <column datatype='boolean' name='[Calculation_{{ calc_id1 }}]' role='dimension' type='nominal'>
              <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHO&quot;)' />
            </column>
            <column datatype='boolean' name='[Calculation_{{ calc_id2 }}]' role='dimension' type='nominal'>
              <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHAT&quot;)' />
            </column>
            <column datatype='boolean' name='[Calculation_{{ calc_id3 }}]' role='dimension' type='nominal'>
              <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHEN&quot;)' />
            </column>
            <column datatype='boolean' name='[Calculation_{{ calc_id4 }}]' role='dimension' type='nominal'>
              <calculation class='tableau' formula='CONTAINS([Missing Elements], &quot;WHY&quot;)' />
            </column>
            <column datatype='string' name='[Control ID]' role='dimension' type='nominal' />
            <column datatype='string' name='[Missing Elements]' role='dimension' type='nominal' />
            <column-instance column='[Control ID]' derivation='Count' name='[cnt:Control ID:qk]' pivot='key' type='quantitative' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style>
          <style-rule element='mark'>
            <encoding attr='color' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[sum:Calculation_{{ calc_id1 }}:qk]' palette='red_blue_diverging' type='interpolated' />
          </style-rule>
        </style>
      </table>
      <simple-id uuid='{{ uuid4 }}' />
    </worksheet>

    <!-- KPI Cards -->
    <worksheet name='KPI Missing WHO'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Controls Missing WHO</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='Control Analyzer Results' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' />
          </datasources>
          <datasource-dependencies datasource='federated.{{ datasource_id }}.{{ datasource_id2 }}'>
            <column datatype='real' name='[Calculation_{{ calc_id8 }}]' role='measure' type='quantitative'>
              <calculation class='tableau' formula='{FIXED : AVG(INT(CONTAINS([Missing Elements], &quot;WHO&quot;)))}' />
            </column>
            <column datatype='string' name='[Missing Elements]' role='dimension' type='nominal' />
            <column-instance column='[Calculation_{{ calc_id8 }}]' derivation='None' name='[none:Calculation_{{ calc_id8 }}:qk]' pivot='key' type='quantitative' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style>
          <style-rule element='cell'>
            <format attr='height' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Calculation_{{ calc_id8 }}:qk]' value='100' />
            <format attr='width' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Calculation_{{ calc_id8 }}:qk]' value='150' />
            <format attr='text-align' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Calculation_{{ calc_id8 }}:qk]' value='center' />
            <format attr='text-size' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Calculation_{{ calc_id8 }}:qk]' value='24' />
          </style-rule>
        </style>
      </table>
      <simple-id uuid='{{ uuid5 }}' />
    </worksheet>

    <!-- Control Details Table -->
    <worksheet name='Control Details'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Control Details</run>
          </formatted-text>
        </title>
      </layout-options>
      <table>
        <view>
          <datasources>
            <datasource caption='Control Analyzer Results' name='federated.{{ datasource_id }}.{{ datasource_id2 }}' />
          </datasources>
          <datasource-dependencies datasource='federated.{{ datasource_id }}.{{ datasource_id2 }}'>
            <column datatype='string' name='[Category]' role='dimension' type='nominal' />
            <column datatype='string' name='[Control Description]' role='dimension' type='nominal' />
            <column datatype='string' name='[Control ID]' role='dimension' type='nominal' />
            <column datatype='real' name='[ESCALATION Score]' role='measure' type='quantitative' />
            <column datatype='string' name='[Missing Elements]' role='dimension' type='nominal' />
            <column datatype='string' name='[Multiple Controls]' role='dimension' type='nominal' />
            <column datatype='real' name='[Total Score]' role='measure' type='quantitative' />
            <column datatype='string' name='[Vague Terms]' role='dimension' type='nominal' />
            <column datatype='real' name='[WHAT Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHEN Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHO Score]' role='measure' type='quantitative' />
            <column datatype='real' name='[WHY Score]' role='measure' type='quantitative' />
            <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' pivot='key' type='nominal' />
            <column-instance column='[Control Description]' derivation='None' name='[none:Control Description:nk]' pivot='key' type='nominal' />
            <column-instance column='[Control ID]' derivation='None' name='[none:Control ID:nk]' pivot='key' type='nominal' />
            <column-instance column='[ESCALATION Score]' derivation='None' name='[none:ESCALATION Score:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Missing Elements]' derivation='None' name='[none:Missing Elements:nk]' pivot='key' type='nominal' />
            <column-instance column='[Multiple Controls]' derivation='None' name='[none:Multiple Controls:nk]' pivot='key' type='nominal' />
            <column-instance column='[Total Score]' derivation='None' name='[none:Total Score:qk]' pivot='key' type='quantitative' />
            <column-instance column='[Vague Terms]' derivation='None' name='[none:Vague Terms:nk]' pivot='key' type='nominal' />
            <column-instance column='[WHAT Score]' derivation='None' name='[none:WHAT Score:qk]' pivot='key' type='quantitative' />
            <column-instance column='[WHEN Score]' derivation='None' name='[none:WHEN Score:qk]' pivot='key' type='quantitative' />
            <column-instance column='[WHO Score]' derivation='None' name='[none:WHO Score:qk]' pivot='key' type='quantitative' />
            <column-instance column='[WHY Score]' derivation='None' name='[none:WHY Score:qk]' pivot='key' type='quantitative' />
          </datasource-dependencies>
          <filter class='categorical' column='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Control ID:nk]'>
            <groupfilter count='25' function='limit' user:op='top' />
          </filter>
          <sort class='computed' column='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Total Score:qk]' direction='DESC' using='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Total Score:qk]' />
          <aggregation value='true' />
        </view>
        <style>
          <style-rule element='cell'>
            <format attr='width' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Control ID:nk]' value='100' />
            <format attr='width' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Control Description:nk]' value='300' />
            <format attr='width' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' value='100' />
          </style-rule>
          <style-rule element='mark'>
            <encoding attr='color' field='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' type='palette'>
              <map to='#28a745'>
                <bucket>&quot;Excellent&quot;</bucket>
              </map>
              <map to='#dc3545'>
                <bucket>&quot;Needs Improvement&quot;</bucket>
              </map>
              <map to='#ffc107'>
                <bucket>&quot;Good&quot;</bucket>
              </map>
            </encoding>
          </style-rule>
        </style>
      </table>
      <simple-id uuid='{{ uuid6 }}' />
    </worksheet>
  </worksheets>

  <!-- Define dashboards -->
  <dashboards>
    <!-- Portfolio Overview Dashboard -->
    <dashboard name='Portfolio Overview'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Control Analyzer QA Dashboard - Portfolio Overview</run>
          </formatted-text>
        </title>
      </layout-options>
      <style>
        <style-rule element='dashboard'>
          <format attr='width' value='1000' />
          <format attr='height' value='800' />
          <format attr='background-color' value='#FFFFFF' />
        </style-rule>
      </style>
      <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
      <zones>
        <zone h='100000' id='4' type='layout-basic' w='100000' x='0' y='0'>
          <zone h='98000' id='3' param='vert' type='layout-flow' w='98400' x='800' y='1000'>
            <zone h='13000' id='6' type='title' w='98400' x='800' y='1000' />
            <zone h='14625' id='5' param='horz' type='layout-flow' w='98400' x='800' y='14000'>
              <zone h='14625' id='13' param='filter' type='filter' w='19680' x='800' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='14' param='filter' type='filter' w='19680' x='20480' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='15' param='filter' type='filter' w='19680' x='40160' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='16' param='filter' type='filter' w='19680' x='59840' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='17' param='filter' type='filter' w='19680' x='79520' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
            </zone>
            <zone h='33750' id='8' param='horz' type='layout-flow' w='98400' x='800' y='28625'>
              <zone h='33750' id='10' name='Score Distribution' w='49200' x='800' y='28625' />
              <zone h='33750' id='11' param='vert' type='layout-flow' w='49200' x='50000' y='28625'>
                <zone h='16875' id='21' name='Category Breakdown' w='49200' x='50000' y='28625' />
                <zone h='16875' id='22' name='Element Radar' w='49200' x='50000' y='45500' />
              </zone>
            </zone>
            <zone h='33750' id='9' param='horz' type='layout-flow' w='98400' x='800' y='62375'>
              <zone h='33750' id='12' name='Missing Elements' w='49200' x='800' y='62375' />
              <zone fixed-size='true' h='33750' id='19' name='KPI Missing WHO' w='49200' x='50000' y='62375'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
            </zone>
          </zone>
        </zone>
      </zones>
      <devicelayouts>
        <devicelayout auto-generated='true' name='Phone'>
          <size maxheight='700' minheight='700' sizing-mode='vscroll' />
          <zones>
            <zone h='100000' id='30' type='layout-basic' w='100000' x='0' y='0'>
              <zone h='98000' id='29' param='vert' type='layout-flow' w='98400' x='800' y='1000'>
                <zone h='13000' id='6' type='title' w='98400' x='800' y='1000' />
                <zone h='10313' id='13' param='filter' type='filter' w='98400' x='800' y='14000'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10312' id='14' param='filter' type='filter' w='98400' x='800' y='24313'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10313' id='15' param='filter' type='filter' w='98400' x='800' y='34625'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10312' id='16' param='filter' type='filter' w='98400' x='800' y='44938'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10313' id='17' param='filter' type='filter' w='98400' x='800' y='55250'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='16875' id='10' name='Score Distribution' w='98400' x='800' y='65563' />
                <zone h='16875' id='21' name='Category Breakdown' w='98400' x='800' y='82438' />
                <zone h='16875' id='22' name='Element Radar' w='98400' x='800' y='99313' />
                <zone h='16875' id='12' name='Missing Elements' w='98400' x='800' y='116188' />
                <zone fixed-size='true' h='16875' id='19' name='KPI Missing WHO' w='98400' x='800' y='133063'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
              </zone>
            </zone>
          </zones>
        </devicelayout>
      </devicelayouts>
      <simple-id uuid='{{ uuid7 }}' />
    </dashboard>

    <!-- Drilldown Dashboard -->
    <dashboard name='Control Drilldown'>
      <layout-options>
        <title>
          <formatted-text>
            <run>Control Analyzer QA Dashboard - Drilldown</run>
          </formatted-text>
        </title>
      </layout-options>
      <style>
        <style-rule element='dashboard'>
          <format attr='width' value='1000' />
          <format attr='height' value='800' />
          <format attr='background-color' value='#FFFFFF' />
        </style-rule>
      </style>
      <size maxheight='800' maxwidth='1000' minheight='800' minwidth='1000' />
      <zones>
        <zone h='100000' id='4' type='layout-basic' w='100000' x='0' y='0'>
          <zone h='98000' id='3' param='vert' type='layout-flow' w='98400' x='800' y='1000'>
            <zone h='13000' id='6' type='title' w='98400' x='800' y='1000' />
            <zone h='14625' id='5' param='horz' type='layout-flow' w='98400' x='800' y='14000'>
              <zone h='14625' id='13' param='filter' type='filter' w='19680' x='800' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='14' param='filter' type='filter' w='19680' x='20480' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='15' param='filter' type='filter' w='19680' x='40160' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='16' param='filter' type='filter' w='19680' x='59840' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
              <zone h='14625' id='17' param='filter' type='filter' w='19680' x='79520' y='14000'>
                <zone-style>
                  <format attr='border-color' value='#000000' />
                  <format attr='border-style' value='solid' />
                  <format attr='border-width' value='1' />
                  <format attr='padding' value='4' />
                </zone-style>
              </zone>
            </zone>
            <zone h='70375' id='7' name='Control Details' w='98400' x='800' y='28625' />
          </zone>
        </zone>
      </zones>
      <devicelayouts>
        <devicelayout auto-generated='true' name='Phone'>
          <size maxheight='700' minheight='700' sizing-mode='vscroll' />
          <zones>
            <zone h='100000' id='24' type='layout-basic' w='100000' x='0' y='0'>
              <zone h='98000' id='23' param='vert' type='layout-flow' w='98400' x='800' y='1000'>
                <zone h='13000' id='6' type='title' w='98400' x='800' y='1000' />
                <zone h='10313' id='13' param='filter' type='filter' w='98400' x='800' y='14000'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10312' id='14' param='filter' type='filter' w='98400' x='800' y='24313'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10313' id='15' param='filter' type='filter' w='98400' x='800' y='34625'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10312' id='16' param='filter' type='filter' w='98400' x='800' y='44938'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='10313' id='17' param='filter' type='filter' w='98400' x='800' y='55250'>
                  <zone-style>
                    <format attr='border-color' value='#000000' />
                    <format attr='border-style' value='solid' />
                    <format attr='border-width' value='1' />
                    <format attr='padding' value='4' />
                  </zone-style>
                </zone>
                <zone h='33750' id='7' name='Control Details' w='98400' x='800' y='65563' />
              </zone>
            </zone>
          </zones>
        </devicelayout>
      </devicelayouts>
      <simple-id uuid='{{ uuid8 }}' />
    </dashboard>
  </dashboards>

  <windows source-height='30'>
    <window class='worksheet' maximized='true' name='Score Distribution'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
        </edge>
        <edge name='right'>
          <strip size='160'>
            <card pane-specification-id='0' param='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' type='color' />
          </strip>
        </edge>
      </cards>
      <viewpoint>
        <highlight>
          <color-one-way>
            <field>[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]</field>
          </color-one-way>
        </highlight>
      </viewpoint>
      <simple-id uuid='{{ uuid9 }}' />
    </window>
    <window class='worksheet' name='Category Breakdown'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
        </edge>
        <edge name='right'>
          <strip size='160'>
            <card pane-specification-id='0' param='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' type='color' />
          </strip>
        </edge>
      </cards>
      <simple-id uuid='{{ uuid10 }}' />
    </window>
    <window class='worksheet' name='Element Radar'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
        </edge>
      </cards>
      <simple-id uuid='{{ uuid11 }}' />
    </window>
    <window class='worksheet' name='Missing Elements'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
        </edge>
        <edge name='right'>
          <strip size='160'>
            <card pane-specification-id='0' param='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[sum:Calculation_{{ calc_id1 }}:qk]' type='color' />
          </strip>
        </edge>
      </cards>
      <simple-id uuid='{{ uuid12 }}' />
    </window>
    <window class='worksheet' name='KPI Missing WHO'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
        </edge>
      </cards>
      <simple-id uuid='{{ uuid13 }}' />
    </window>
    <window class='worksheet' name='Control Details'>
      <cards>
        <edge name='left'>
          <strip size='160'>
            <card type='pages' />
            <card type='filters' />
            <card type='marks' />
          </strip>
        </edge>
        <edge name='top'>
          <strip size='31'>
            <card type='columns' />
          </strip>
          <strip size='31'>
            <card type='rows' />
          </strip>
        </edge>
        <edge name='right'>
          <strip size='160'>
            <card pane-specification-id='0' param='[federated.{{ datasource_id }}.{{ datasource_id2 }}].[none:Category:nk]' type='color' />
          </strip>
        </edge>
      </cards>
      <simple-id uuid='{{ uuid14 }}' />
    </window>
    <window class='dashboard' name='Portfolio Overview'>
      <viewpoints>
        <viewpoint name='Category Breakdown'>
          <zoom type='entire-view' />
        </viewpoint>
        <viewpoint name='Element Radar'>
          <zoom type='entire-view' />
        </viewpoint>
        <viewpoint name='KPI Missing WHO'>
          <zoom type='entire-view' />
        </viewpoint>
        <viewpoint name='Missing Elements'>
          <zoom type='entire-view' />
        </viewpoint>
        <viewpoint name='Score Distribution'>
          <zoom type='entire-view' />
        </viewpoint>
      </viewpoints>
      <active id='19' />
      <simple-id uuid='{{ uuid15 }}' />
    </window>
    <window class='dashboard' maximized='true' name='Control Drilldown'>
      <viewpoints>
        <viewpoint name='Control Details'>
          <zoom type='entire-view' />
        </viewpoint>
      </viewpoints>
      <active id='-1' />
      <simple-id uuid='{{ uuid16 }}' />
    </window>
  </windows>
  <thumbnails>
    <thumbnail height='192' name='Score Distribution' width='192'>
      THUMBNAIL_PLACEHOLDER_1
    </thumbnail>
    <thumbnail height='192' name='Category Breakdown' width='192'>
      THUMBNAIL_PLACEHOLDER_2
    </thumbnail>
    <thumbnail height='192' name='Element Radar' width='192'>
      THUMBNAIL_PLACEHOLDER_3
    </thumbnail>
    <thumbnail height='192' name='Missing Elements' width='192'>
      THUMBNAIL_PLACEHOLDER_4
    </thumbnail>
    <thumbnail height='192' name='KPI Missing WHO' width='192'>
      THUMBNAIL_PLACEHOLDER_5
    </thumbnail>
    <thumbnail height='192' name='Control Details' width='192'>
      THUMBNAIL_PLACEHOLDER_6
    </thumbnail>
    <thumbnail height='192' name='Portfolio Overview' width='192'>
      THUMBNAIL_PLACEHOLDER_7
    </thumbnail>
    <thumbnail height='192' name='Control Drilldown' width='192'>
      THUMBNAIL_PLACEHOLDER_8
    </thumbnail>
  </thumbnails>
</workbook>
"""


def generate_twb_file(output_path="Control_Analyzer_Dashboard.twb"):
    """
    Generate a Tableau workbook (.twb) file using the template

    Args:
        output_path: Path to save the generated .twb file
    """
    # Create UUIDs for all components
    template_vars = {
        'datasource_id': str(uuid.uuid4()).replace('-', '_'),
        'datasource_id2': str(uuid.uuid4()).replace('-', '_'),
        'connection_id': str(uuid.uuid4()).replace('-', '_'),
        'object_id1': str(uuid.uuid4()).replace('-', '_'),
        'calc_id1': str(uuid.uuid4()).replace('-', '_'),
        'calc_id2': str(uuid.uuid4()).replace('-', '_'),
        'calc_id3': str(uuid.uuid4()).replace('-', '_'),
        'calc_id4': str(uuid.uuid4()).replace('-', '_'),
        'calc_id5': str(uuid.uuid4()).replace('-', '_'),
        'calc_id6': str(uuid.uuid4()).replace('-', '_'),
        'calc_id7': str(uuid.uuid4()).replace('-', '_'),
        'calc_id8': str(uuid.uuid4()).replace('-', '_'),
        'calc_id9': str(uuid.uuid4()).replace('-', '_'),
        'calc_id10': str(uuid.uuid4()).replace('-', '_'),
        'calc_id11': str(uuid.uuid4()).replace('-', '_'),
        'calc_id12': str(uuid.uuid4()).replace('-', '_'),
        'calc_id13': str(uuid.uuid4()).replace('-', '_'),
        'uuid1': str(uuid.uuid4()),
        'uuid2': str(uuid.uuid4()),
        'uuid3': str(uuid.uuid4()),
        'uuid4': str(uuid.uuid4()),
        'uuid5': str(uuid.uuid4()),
        'uuid6': str(uuid.uuid4()),
        'uuid7': str(uuid.uuid4()),
        'uuid8': str(uuid.uuid4()),
        'uuid9': str(uuid.uuid4()),
        'uuid10': str(uuid.uuid4()),
        'uuid11': str(uuid.uuid4()),
        'uuid12': str(uuid.uuid4()),
        'uuid13': str(uuid.uuid4()),
        'uuid14': str(uuid.uuid4()),
        'uuid15': str(uuid.uuid4()),
        'uuid16': str(uuid.uuid4()),
    }

    # Create template from the TWB_TEMPLATE
    template = Template(TWB_TEMPLATE)

    # Render the template with the variables
    twb_content = template.render(**template_vars)

    # Format XML for readability (optional)
    try:
        dom = xml.dom.minidom.parseString(twb_content)
        pretty_xml = dom.toprettyxml(indent="  ")
        # Remove extra whitespace that parseString can introduce
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
        twb_content = pretty_xml
    except Exception as e:
        print(f"Warning: Could not format XML: {e}")

    # Write the TWB file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(twb_content)

    print(f"Tableau workbook created successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate the Tableau workbook
    generate_twb_file()