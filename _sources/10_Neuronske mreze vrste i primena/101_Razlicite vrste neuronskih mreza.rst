Учење репрезентације података
===============================

.. infonote::

 Неуронске мреже нам могу помоћи да издвојимо неке апстрактне атрибуте у подацима и научимо репрезентације које су подесне за решавање задатака. 


У примерима које смо до сада користили најчешће смо се ослањали на постојање неког скупа атрибута у скупу података. Уистину говорећи, велики
број домена генерише баш податке који су овог облика, са атрибутима у колонама и инстанцама у појединачним редовима. Као што смо видели у 
уводном делу у причи о припреми података, чак и када располажемо оваквим атрибутима није баш најинтуитивније одлучити које атрибуте треба да 
одаберемо за креирање модела. То нас је стављало у позицију да опробавамо различите комбинације или осмишљавамо технике које нам могу помоћи у 
селекцији атрибута. Услед комплексности функција које моделују, неуронске мреже се могу похвалити својством да могу лепо да науче да филтрирају и 
групишу атрибуте који су битни. 

|

Ово својство неуронских мрежа је посебно важно у раду са подацима који нису табеларни - покренули смо већ много пута питање како представити, 
на пример, слике, текстуалне податке или аудио-записе. Иако имамо знања о овим форматима, тешко нам је да опишемо шта тачно они садрже на неки 
концизан и употребљив начин. То нас је, између осталог, и мотивисало за примену парадигме програмирања вођеног подацима. Неуронске мреже могу 
(у то ћемо се ускоро уверити) на основу података у изворном облику да науче неке апстрактне атрибуте који су корисни за успешно решавање задатака.

|

У наставку ћемо упознати конволутивне неуронске мреже које се користе примарно у раду са сликама и видеом и за учење визуелних атрибута улаза, а 
затим и рекурентне неуронске мреже и трансформере, типове неуронских мрежа које се користе за учење атрибута секвенцијалних података као што су текст или звук. 

